from concurrent import futures
from threading import Event

import grpc
import signal

import nltk

from mir.config.config import Config
from mir.config import settings
from mir.search.common import load_phase1, load_phase2, load_phase3, load_similar_papers, load_languages, DependentRunner
from mir.util import getLogger

logger = getLogger(__name__)


class GracefulKiller:
    kill_now = False

    def __init__(self):
        self.__event = Event()
        signal.signal(signal.SIGINT, self.__exit_gracefully)
        signal.signal(signal.SIGTERM, self.__exit_gracefully)

    def __exit_gracefully(self, signum, frame):
        self.__event.set()

    def wait(self):
        self.__event.wait()


def serve(config: Config):
    logger.info('starting server')

    killer = GracefulKiller()

    executor = futures.ThreadPoolExecutor(max_workers=config.grpc.num_workers)

    languages = load_languages(executor)
    with DependentRunner.use_default_dependencies(*languages):
        ai_bio, ai_bio_nodes = load_phase1(executor, 'ai-bio')
        hw_system, hw_system_nodes = load_phase1(executor, 'hardware-system')
        arxiv, arxiv_nodes = load_phase2(executor)
        ss, ss_nodes = load_phase3(executor)
        similar_papers, similar_papers_nodes = load_similar_papers(executor)
        DependentRunner(lambda: logger.info('all dependencies are loaded'),
                        *ai_bio_nodes, *hw_system_nodes, *arxiv_nodes, *ss_nodes, *similar_papers_nodes) \
            .submit_to(executor)
    server = grpc.server(executor)
    settings.SERVICER_ADDER(settings.SERVICE(ai_bio, hw_system, arxiv, ss, similar_papers), server)

    listen_addr = f'[::]:{config.grpc.listen_port}'
    server.add_insecure_port(listen_addr)
    server.start()

    logger.info(f'started server on {listen_addr}')

    killer.wait()
    logger.info('stopping server')
    server.stop(0)
