import logging
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
logger = logging.getLogger('day22')
class ServiceError(Exception):
    def __init__(self, status_code: int, error: str, detail: str, headers: dict | None = None):
        self.status_code = status_code
        self.error = error
        self.detail = detail
        self.headers = headers
def build_response(request: Request, status_code: int, error: str, detail: str, headers: dict | None = None):
    request_id = getattr(request.state, 'request_id', 'unknown')
    body = {'request_id': request_id, 'error': error, 'detail': detail}
    return JSONResponse(status_code=status_code, content=body, headers=headers)
async def service_error_handler(request: Request, exc: ServiceError):
    return build_response(request, exc.status_code, exc.error, exc.detail, exc.headers)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    first = exc.errors()[0]
    where = '.'.join(str(part) for part in first['loc'])
    message = first['msg']
    return build_response(request, 422, 'invalid_request', f'{where}: {message}')
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception('Unhandled error on %s', request.url.path)
    return build_response(request, 500, 'internal_error', 'Something broke on our side. Use the request_id when you report it.')
