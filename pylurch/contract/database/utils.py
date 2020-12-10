from functools import wraps


# TODO: Workaround: https://github.com/tiangolo/pydantic-sqlalchemy/issues/10
def custom_column_property(f, key):
    @wraps(f)
    def wrapper(*args, default=None, nullable=True, **kwargs):
        v = f(*args, **kwargs)
        for c in v.columns:
            c.default = default
            c.nullable = nullable
            # TODO: Perhaps ok...?
            c.key = c.name = key

        return v

    return wrapper
