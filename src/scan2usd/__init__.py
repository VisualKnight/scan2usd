__version__ = "0.1.0"


def _register_schemas() -> None:
    try:
        from .schemas import register

        register()
    except Exception:  # never block imports; files stay readable without the schema plugin
        pass


_register_schemas()
