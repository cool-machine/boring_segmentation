from azure.ai.ml import MLClient
from azure.ai.ml.entities import JobSchedule, CronTrigger
from azure.identity import DefaultAzureCredential

# Authenticate
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# Define a job schedule
schedule = JobSchedule(
    name="daily-training-schedule",
    display_name="Daily Training Job",
    trigger=CronTrigger(expression="0 6 * * *", time_zone="UTC"),  # Runs daily at 6 AM UTC
    job="./src/scheduling/train_schedule.yml"
)

# Create the schedule
ml_client.schedules.create_or_update(schedule)
print("✅ Training job scheduled!")
