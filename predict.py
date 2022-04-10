import os
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Optional, Union
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BeamSearchScorer,
)
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)
from transformers.generation_utils import (
    BeamSearchEncoderDecoderOutput,
    BeamSearchDecoderOnlyOutput,
    BeamSearchOutput,
)

torch.set_num_threads(1)

def get_logits_processor(
        config,
        encoder_input_ids: torch.LongTensor,
        min_length: int,
        max_length: int,
        num_beams: int,
):  
    processors = LogitsProcessorList()
    repetition_penalty = config.repetition_penalty
    no_repeat_ngram_size = (config.no_repeat_ngram_size)
    encoder_no_repeat_ngram_size = (config.encoder_no_repeat_ngram_size)
    bad_words_ids = config.bad_words_ids
    eos_token_id = config.eos_token_id
    diversity_penalty = config.diversity_penalty
    num_beam_groups = config.num_beam_groups
    forced_bos_token_id = (config.forced_bos_token_id)
    forced_eos_token_id = (config.forced_eos_token_id)
    remove_invalid_values = (config.remove_invalid_values)

    if diversity_penalty is not None and diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
            )
        )
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0: 
        processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if encoder_no_repeat_ngram_size is not None and encoder_no_repeat_ngram_size > 0:
        processors.append(EncoderNoRepeatNGramLogitsProcessor(encoder_no_repeat_ngram_size, encoder_input_ids))

    if bad_words_ids is not None:
        processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
    # we use 1 for min_length
    if min_length is not None and eos_token_id is not None and min_length > -1:
        processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
    if forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
    # default forced_eos_token_id is 2
    if forced_eos_token_id is not None:
        processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
    if remove_invalid_values is True:
        processors.append(InfNanRemoveLogitsProcessor())
    return processors

def beam_search(
    model,
    input_ids: torch.LongTensor,
    beam_scorer,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    pad_token_id: int,
    eos_token_id: int,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    if len(stopping_criteria) == 0:
        print("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    output_scores = output_scores if output_scores is not None else model.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    assert (
        num_beams * batch_size == batch_beam_size
    ), f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    while True:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        next_token_logits = outputs.logits[:, -1, :]
        next_token_logits = next_token_logits
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

        next_token_scores = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
        if model_kwargs["past"] is not None:
            model_kwargs["past"] = model._reorder_cache(model_kwargs["past"], beam_idx)

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):     
            break
  
    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None
        return BeamSearchEncoderDecoderOutput(
            sequences=sequence_outputs["sequences"],
            sequences_scores=sequence_outputs["sequence_scores"],
            scores=scores,
            encoder_attentions=encoder_attentions,
            encoder_hidden_states=encoder_hidden_states,
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
            decoder_hidden_states=decoder_hidden_states,
        )
    else:
        return sequence_outputs["sequences"]

def generate_beam_search(
        text: str,
        model,
        tokenizer,
        loss_fn=CrossEntropyLoss(),
        vocab_text: str = None,
        num_beams: int = 4,
        max_source_length: int = 2048,
        min_target_length: int = 20,
        max_target_length: int = 400,
        control: str=None,
        repeat_control: int=1,
):
    device = model.device
    model.eval()
    
    encoder_input_ids = tokenizer(text,max_length=max_source_length,padding=False,truncation=True,return_tensors="pt").input_ids.to(device)
    
    encoder_outputs = model.get_encoder()(encoder_input_ids, return_dict=True)
    expanded_return_idx = (torch.arange(encoder_input_ids.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1))
    # instead of copy all states, only copy the hidden_states to save gpu space
    encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(0, expanded_return_idx.to(device))
    model_kwargs = {"encoder_outputs": encoder_outputs,}
    eos, bos = tokenizer.eos_token_id, tokenizer.bos_token_id

    assert model.config.model_type == "bart"
    decoder_input_ids_base = torch.LongTensor([[model.config.decoder_start_token_id]]).to(device)

    # for original
    min_length = min_target_length
    max_length = max_target_length
    logits_processor = get_logits_processor(model.config, encoder_input_ids = encoder_input_ids, min_length = min_length, max_length = max_length,num_beams = num_beams)
    length_penalty = model.config.length_penalty # default is 2
    early_stopping = model.config.early_stopping # true
    num_return_sequences = model.config.num_return_sequences
    beam_scorer = BeamSearchScorer(batch_size=1,num_beams=num_beams,device=device,length_penalty=length_penalty,do_early_stopping=early_stopping,num_beam_hyps_to_keep=num_return_sequences)
    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(MaxLengthCriteria(max_length=max_length))

    with torch.no_grad():
        decoder_input_ids = decoder_input_ids_base.index_select(0, expanded_return_idx.to(device))
        outputs = beam_search(
            model,
            decoder_input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=model.config.pad_token_id if model.config.pad_token_id is not None else eos,
            eos_token_id=eos,
            return_dict_in_generate=model.config.return_dict_in_generate,
            **model_kwargs,
        )
        if isinstance(outputs, BeamSearchEncoderDecoderOutput) or isinstance(outputs, BeamSearchDecoderOnlyOutput):
            outputs = outputs.sequences
        assert outputs.dim() == 2 and outputs.size(0) == 1

    text = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
    return text


class predict_mred(object):

    def init(self, model_dir="./model"):
        gpu_avail = torch.cuda.is_available()
        if gpu_avail:
            device = torch.device('cuda')
        else:
            device = torch.device("cpu")
        print("using device:", device)
        print('==================init start=====================')
        # load model
        model_dir = "./model"
        self.lm = AutoModelForSeq2SeqLM.from_pretrained(model_dir, return_dict_in_generate=False).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

        # 文件修改
        self.dynamic_time = -1
        self.dynamic_file = None
        print('===================init end======================')

    def process(self, data):
        input_text = data["data"]
        output_text = generate_beam_search(input_text, self.lm, self.tokenizer)
        outputs = dict()
        outputs["result"] = output_text
        outputs["dynamic"] = self.dynamic_file
        return outputs

    def update(self, data_dir):
        update_file = "%s/update_file" % data_dir
        timestamp = os.path.getmtime(update_file)
        if timestamp > self.dynamic_time:
            with open(update_file, "r") as f:
                self.dynamic_file = f.readlines()
            self.dynamic_time = timestamp

if __name__ == "__main__":
    import json
    with open("body.txt", "r", encoding="utf-8") as data_file:
        data = json.load(data_file)
    pm = predict_mred()

    print("init model")
    pm.init()

    print(pm.process(data))