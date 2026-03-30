import sys
sys.path.insert(0, '.')
import routes.predict as rp
import inspect
src = inspect.getsource(rp.run_prediction)
print('HAS_ENTROPY_DEBUG:', 'ENTROPY DEBUG' in src)
print('HAS_LOGIT_NORM:', 'LOGIT NORMALIZATION' in src)
print('HAS_ENTROPY_OUTPUT:', 'ENTROPY OUTPUT' in src)
print('HAS_UNCERTAINTY_PROFILE:', 'UNCERTAINTY PROFILE' in src)
print('\nFirst 300 chars:')
print(src[:300])
