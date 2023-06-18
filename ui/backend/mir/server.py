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


ai_bio = None
hw_system = None
arxiv = None
ss = None
similar_papers = None


def set_global(ai_bio_: Phase1, hw_system_: Phase1, arxiv_: Phase2, ss_: Phase3, similar_papers_: SimilarPapers):
    global ai_bio, hw_system, arxiv, ss, similar_papers
    ai_bio = ai_bio_
    hw_system = hw_system_
    arxiv = arxiv_
    ss = ss_
    similar_papers = similar_papers_


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
    set_global(ai_bio, hw_system, arxiv, ss, similar_papers)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.grpc.num_workers))
                                                     # initializer=set_global,
                                                     # initargs=(ai_bio, hw_system, arxiv, ss, similar_papers)))
    settings.SERVICER_ADDER(settings.SERVICE(), server)

    listen_addr = f'[::]:{config.grpc.listen_port}'
    server.add_insecure_port(listen_addr)
    server.start()

    logger.info(f'started server on {listen_addr}')

    killer.wait()
    logger.info('stopping server')
    server.stop(0)
