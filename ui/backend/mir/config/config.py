from dataclasses import dataclass


class BaseConfig:

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                attr = getattr(self, k)
                if isinstance(v, dict):
                    assert attr is None or isinstance(attr, (BaseConfig, dict)), \
                        f"Expected {k} to be a dict, BaseConfig or None, got {type(attr)}"
                    attr.update(**v)
                else:
                    setattr(self, k, v)
            else:
                raise AttributeError(f"Invalid attribute {k}")


@dataclass
class GrpcConfig(BaseConfig):
    listen_port: int = 50051
    num_workers: int = 4


@dataclass
class Config(BaseConfig):
    config_file: str = None
    grpc: GrpcConfig = GrpcConfig()
