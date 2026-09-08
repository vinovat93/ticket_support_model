import argparse

from model import Model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--to_train",
        action="store_true",
        help="Train the model and save it before predicting (default: skip training, use the saved model).",
    )
    parser.add_argument(
        "--to_calibrate",
        action="store_true",
        help="Here we use GridSearchCV to find the best arguments calibration for our model..",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model = Model()

    if args.to_calibrate:
        model.parameter_calibration()

    if args.to_train:
        model.train()

    tech_support_scenarios = [
        "Hi team, the ERP system keeps rejecting my purchase order with a validation error every time I try to submit it. Can you take a look?",
        "Hey, my laptop's Windows update has been stuck installing for two days and I can't use it. Can you help?",
        "Hi team, I got an alert that someone signed into my account from Romania at 3am, this looks like account takeover. Can you take a look?",
        "Hello, my email account says it's locked and I can't get in. Also, a distribution list lost access for half the team. Appreciate any help.",
        "Hi, my security group membership isn't giving me access to files I'm supposed to see, and I need admin rights for the deployment tool. Let me know what's needed.",
        "Hey, my laptop won't turn on this morning, nothing happens at all. This is blocking my work.",
        "Hi team, my voicemail has terrible audio quality and the call queue has no dial tone. Can you take a look?",
        "Good morning, the connection pool keeps returning 'too many connections' and everything is really slow today. Appreciate any help.",
        "Hello, internet access keeps dropping every few minutes and I can't reach anything internal since this morning. Please advise.",
        "Hi team, the production EC2 instance is running at 100% CPU and the autoscaling group won't scale up under load. Can you take a look?",
    ]

    results = model.predict(tech_support_scenarios)

    print(results.to_string(index=False))
