import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from deal_pilot.app.experiment_webhook import router as experiment_router
from deal_pilot.app.alert_webhook import router as alert_router
from deal_pilot.event.listener import cancel_feishu_events, start_feishu_events

@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_feishu_events()

    try:
        yield
    finally:
        await asyncio.to_thread(cancel_feishu_events)

app = FastAPI(lifespan=lifespan)
app.include_router(experiment_router)
app.include_router(alert_router)
