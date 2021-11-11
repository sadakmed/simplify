# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from pathlib import Path

import re
import shutil
import shlex


from fairseq_cli import generate


from .utils import (
    log_std_streams,
    yield_lines,
    write_lines,
    mock_cli_args,
    create_temp_dir,
    mute,
    args_dict_to_str,
)


def remove_multiple_whitespaces(text):
    return re.sub(r"  +", " ", text)


def fairseq_parse_all_hypotheses(out_filepath):
    hypotheses_dict = defaultdict(list)
    for line in yield_lines(out_filepath):
        match = re.match(r"^H-(\d+)\t-?\d+\.\d+\t(.*)$", line)
        if match:
            sample_id, hypothesis = match.groups()
            hypotheses_dict[int(sample_id)].append(hypothesis)
    # Sort in original order
    return [hypotheses_dict[i] for i in range(len(hypotheses_dict))]


def _fairseq_generate(
    complex_filepath,
    output_pred_filepath,
    checkpoint_paths,
    complex_dictionary_path,
    simple_dictionary_path,
    beam=5,
    hypothesis_num=1,
    lenpen=1.0,
    diverse_beam_groups=None,
    diverse_beam_strength=0.5,
    sampling=False,
    max_tokens=16384,
    source_lang="complex",
    target_lang="simple",
    **kwargs,
):
    # exp_dir must contain checkpoints/checkpoint_best.pt, and dict.{complex,simple}.txt
    # First copy input complex file to exp_dir and create dummy simple file

    with create_temp_dir() as temp_dir:
        new_complex_filepath = (
            temp_dir / f"tmp.{source_lang}-{target_lang}.{source_lang}"
        )
        dummy_simple_filepath = (
            temp_dir / f"tmp.{source_lang}-{target_lang}.{target_lang}"
        )
        shutil.copy(complex_filepath, new_complex_filepath)
        shutil.copy(complex_filepath, dummy_simple_filepath)
        shutil.copy(complex_dictionary_path, temp_dir / f"dict.{source_lang}.txt")
        shutil.copy(simple_dictionary_path, temp_dir / f"dict.{target_lang}.txt")
        args = f"""
        {temp_dir} --dataset-impl raw --gen-subset tmp --path {':'.join([str(path) for path in checkpoint_paths])}
        --beam {beam} --nbest {hypothesis_num} --lenpen {lenpen}
        --diverse-beam-groups {diverse_beam_groups if diverse_beam_groups is not None else -1} --diverse-beam-strength {diverse_beam_strength}
        --max-tokens {max_tokens}
        --model-overrides "{{'encoder_embed_path': None, 'decoder_embed_path': None}}"
        --skip-invalid-size-inputs-valid-test
        """
        if sampling:
            args += f"--sampling --sampling-topk 10"
        # FIXME: if the kwargs are already present in the args string, they will appear twice but fairseq will take only the last one into account
        args += f" {args_dict_to_str(kwargs)}"
        args = remove_multiple_whitespaces(args.replace("\n", " "))
        out_filepath = temp_dir / "generation.out"
        with mute(mute_stderr=False):
            with log_std_streams(out_filepath):
                # evaluate model in batch mode
                args = shlex.split(args)
                with mock_cli_args(args):
                    generate.cli_main()

        all_hypotheses = fairseq_parse_all_hypotheses(out_filepath)
        predictions = [hypotheses[hypothesis_num - 1] for hypotheses in all_hypotheses]
        write_lines(predictions, output_pred_filepath)


def fairseq_generate(
    complex_filepath,
    output_pred_filepath,
    exp_dir,
    beam=5,
    hypothesis_num=1,
    lenpen=1.0,
    diverse_beam_groups=None,
    diverse_beam_strength=0.5,
    sampling=False,
    max_tokens=8000,
    source_lang="complex",
    target_lang="simple",
    **kwargs,
):

    exp_dir = Path(exp_dir)
    possible_checkpoint_paths = [
        exp_dir / "model.pt",
        exp_dir / "checkpoints/checkpoint_best.pt",
        exp_dir / "checkpoints/checkpoint_last.pt",
    ]
    assert any(
        [path for path in possible_checkpoint_paths if path.exists()]
    ), f"Generation failed, no checkpoint found in {possible_checkpoint_paths}"  # noqa: E501
    checkpoint_path = [path for path in possible_checkpoint_paths if path.exists()][0]
    complex_dictionary_path = exp_dir / f"dict.{source_lang}.txt"
    simple_dictionary_path = exp_dir / f"dict.{target_lang}.txt"
    _fairseq_generate(
        complex_filepath,
        output_pred_filepath,
        [checkpoint_path],
        complex_dictionary_path=complex_dictionary_path,
        simple_dictionary_path=simple_dictionary_path,
        beam=beam,
        hypothesis_num=hypothesis_num,
        lenpen=lenpen,
        diverse_beam_groups=diverse_beam_groups,
        diverse_beam_strength=diverse_beam_strength,
        sampling=sampling,
        max_tokens=max_tokens,
        **kwargs,
    )
