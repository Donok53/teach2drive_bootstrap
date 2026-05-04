"""Rule-aware token training entry point.

This keeps the original token imitation trainer available for the baseline track
while exposing stronger stop-state and stop-reason supervision for comparison.
"""

from .train_tokens import build_arg_parser, train


def main():
    parser = build_arg_parser()
    parser.description = "Train a rule-aware token-fusion policy with stop-state and stop-reason heads."
    parser.set_defaults(
        speed_loss_weight=0.35,
        stop_loss_weight=0.05,
        control_loss_weight=0.25,
        lane_loss_weight=0.15,
        stop_state_loss_weight=0.35,
        stop_reason_loss_weight=0.15,
        camera_dropout=0.05,
        lidar_dropout=0.05,
        step_log_every=200,
        log_every=1,
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
