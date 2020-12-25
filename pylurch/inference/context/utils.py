def container_exists(train_container=True):
    attribute = "_container" if train_container else "_loaded_container"

    def wrapper2(f):
        def wrapper(self, *args, **kwargs):
            if getattr(self, attribute, None) is None:
                msg = "No model created/fitted!" if train_container else "No model loaded!"
                raise Exception(msg)

            return f(self, *args, **kwargs)

        return wrapper

    return wrapper2