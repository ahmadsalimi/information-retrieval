try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution('mir').version
except pkg_resources.DistributionNotFound:
    __version__ = 'local'
