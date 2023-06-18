from concurrent import futures
from threading import Event

import grpc
import signal

import nltk

from mir.config.config import Config
from mir.config import settings
from mir.search.common import load_phase1, load_phase2, load_phase3, load_similar_papers, Phase1, Phase2, Phase3, \
    SimilarPapers
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

    nltk.download('punkt')
    nltk.download('stopwords')
    ai_bio = load_phase1('ai-bio')
    hw_system = load_phase1('hardware-system')
    arxiv = load_phase2()
    ss = load_phase3()
    similar_papers = load_similar_papers()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.grpc.num_workers))
    settings.SERVICER_ADDER(settings.SERVICE(ai_bio, hw_system, arxiv, ss, similar_papers), server)

    listen_addr = f'[::]:{config.grpc.listen_port}'
    server.add_insecure_port(listen_addr)
    server.start()

    logger.info(f'started server on {listen_addr}')

    killer.wait()
    logger.info('stopping server')
    server.stop(0)
