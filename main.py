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
        "A user reports that a specific API endpoint suddenly returns a 500 Internal Server Error whenever they send a payload containing special characters, blocking their third-party integration.",
        "An analyst complains that the daily analytics dashboard is showing stale data because the scheduled ETL pipeline failed overnight due to a schema mismatch in the source database.",
        "A developer states that their deployment to the staging environment is stuck in a loop because the CI/CD pipeline is failing at the Docker image build stage due to an expired dependency mirror URL.",
        "A user reports that after clicking the 'Checkout' button, the screen goes completely white, and the browser console throws an unhandled runtime error: 'Cannot read properties of undefined (reading \"id\")'.",
        "A data scientist complains that they cannot deploy their trained model to production because the model registry throws an 'Out of Memory' error during the serialization check.",
        "An Android user reports that the app immediately crashes on launch right after upgrading to the latest version, while iOS users are unaffected.",
        "An automated integration test suite is intermittently failing on the main branch, but the issue cannot be replicated locally on individual developer machines (flaky test scenario).",
        "An automated alert is triggered because the application's response latency has spiked to over 5 seconds for 95% of users, indicating a potential connection pool exhaustion.",
        "An employee reports receiving an automated security notification stating that an unauthorized API token associated with their account was leaked in a public GitHub repository.",
        "A customer calls frustrated because they are locked out of their account, and the 'Reset Password' link sent to their email keeps returning an 'Invalid or Expired Token' error."
    ]

    results = model.predict(tech_support_scenarios)

    print(results.to_string(index=False))
