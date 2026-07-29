import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from state_ai.model import Student
from state_ai.train_kd import load_soft_labels, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="artifacts/teacher_soft_labels_train_v5.json")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default="artifacts/student_model_ckpt")
    args = p.parse_args()

    if not Path(args.labels).exists():
        sys.exit(f"missing {args.labels}")

    states = load_soft_labels(args.labels)
    print(len(states), "samples")
    student = Student()
    if Path(args.out).exists():
        print("warm start", args.out)
        student.load(args.out)
    train(student, states, epochs=args.epochs, batch_size=args.batch_size)
    student.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
