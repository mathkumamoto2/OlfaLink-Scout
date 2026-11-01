"""Desktop command plumbing tests do not require Genesis hardware."""
import pytest
from pathlib import Path
from scout_genesis.cli import (
    make_parser, select_demo_policy, summary_of, scripted_action, action_for,
)
import numpy as np


def test_auto_policy_no_model_is_scripted(tmp_path):
    assert select_demo_policy('auto', tmp_path / 'missing.zip') == 'scripted'
    f = tmp_path / 'ppo_genesis.zip'
    f.write_bytes(b'mock')
    assert select_demo_policy('auto', f) == 'ppo'
    assert select_demo_policy('random', f) == 'random'


def test_scripted_is_not_random():
    seq_a = [scripted_action(i) for i in range(80)]
    seq_b = [scripted_action(i) for i in range(80)]
    assert seq_a == seq_b
    assert set(seq_a) == {1, 2, 3}
    assert action_for('scripted', np.zeros(15), None, np.random.default_rng(5), 5, False) == 1


def test_summary_uses_actual_values():
    rows = [
        dict(policy='ppo', success=1, steps=18, distance_to_source_m=.1),
        dict(policy='ppo', success=0, steps=160, distance_to_source_m=.5),
        dict(policy='random', success=0, steps=160, distance_to_source_m=1.4),
    ]
    out = summary_of(rows)
    assert out['ppo']['success_rate'] == pytest.approx(.5)
    assert out['ppo']['mean_steps'] == 89
    assert out['random']['success_rate'] == 0


def test_cli_arguments_windows_defaults():
    args = make_parser().parse_args(['demo', '--backend', 'cpu', '--policy', 'ppo', '--record'])
    assert args.record and args.policy == 'ppo'
    assert args.video_actions * 8 * .02 >= 60
    assert args.output == Path('outputs')
