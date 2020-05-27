class BaseInterface(object):
    def __init__(self, base: str, endpoint: str):
        """
        Defines a base class for interfaces.
        :param base: The base address of the server
        :param endpoint: The endpoint of the server
        """

        self._base = base if not base.endswith('/') else base[:-1]

        if endpoint.startswith('/'):
            raise ValueError('The endpoint should not begin with a `/`!')

        self._ep = endpoint

        self._headers = {
            'Content-type': 'application/json'
        }

    def url(self):
        return f'{self._base}/{self._ep}'

    def _exec_req(self, meth, **kwargs):
        resp = meth(self.url(), headers=self._headers, **kwargs)

        if resp.status_code != 200:
            raise Exception(f'Got error code {resp.status_code}: {resp.text}')

        return resp.json()

    def add_header(self, k: str, v: str):
        """
        Add headers to request.
        :param k: Key
        :param v: Value
        """

        self._headers[k] = v
        return self