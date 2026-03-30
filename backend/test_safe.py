import sys, asyncio, logging
sys.path.insert(0, '.')

# Suppress UnicodeEncodeError from emoji in log messages
class SafeHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            super().emit(record)
        except (UnicodeEncodeError, Exception):
            pass

logging.basicConfig(handlers=[SafeHandler()], level=logging.WARNING)

async def test():
    from services.predictor import run_prediction
    with open('../Testing/pituitary/1201.jpg', 'rb') as f:
        b = f.read()
    r = await run_prediction(b)
    print('tumor_type        :', r.get('tumor_type'))
    print('prediction_entropy:', r.get('prediction_entropy'))
    print('uncertainty_profile:', r.get('uncertainty_profile'))
    print('error             :', r.get('error'))

asyncio.run(test())
