import logging
import random
from typing import Dict, Any

from melissa.server.sensitivity_analysis import SensitivityAnalysisServer
from melissa.launcher import message
from melissa.scheduler import job

logger = logging.getLogger("melissa")


class HeatPDEServerSA(SensitivityAnalysisServer):
    """
    Use-case specific server
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        random.seed(self.study_options['seed'])

    def draw_parameters(self):
        mesh_size = self.study_options['mesh_size']
        time_discretization = self.study_options['time_discretization']
        Tmin, Tmax = self.study_options['parameter_range']
        param_set = [mesh_size, mesh_size, time_discretization]
        for i in range(self.study_options['nb_parameters']):
            param_set.append(random.uniform(Tmin, Tmax))
        return param_set

    def start(self):
        """
        The main execution method
        """
        self.launch_first_groups()
        self.receive()
        logger.info("stop server")
        self.close_connection()

    def handle_fd(self):
        """
        Handles the launcher's messages through the filedescriptor
        """
        bs = self.launcherfd.recv(256)
        rcvd_msg = self.decode_msg(bs)

        for msg in rcvd_msg:
            # 1. Launcher sent JOB_UPDATE message (msg.job_id <=> group_id)
            if isinstance(msg, message.JobUpdate):

                # React to simulation status
                if msg.job_state == job.State.TERMINATED:
                    logger.debug(f"Launcher indicates job termination (job {msg.job_id})")
                    self.n_finished_simulations += 1

        # 2. Server sends PING
        logger.debug("Server got message from launcher and sends PING back")
        snd_msg = self.encode_msg(message.Ping())
        self.launcherfd.send(snd_msg)
