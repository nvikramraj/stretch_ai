#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import click

# import stretch.utils.logger as logger
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.pickup import PickupExecutor
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.llms import LLMChatWrapper, PickupPromptBuilder, get_llm_choices, get_llm_client
from stretch.perception import create_semantic_sensor
from stretch.utils.logger import Logger
from termcolor import colored
from stretch.agent.base import ManagedOperation
import time
import gc
import torch
import stretch.core.status as status
from stretch.motion import HelloStretchIdx
import numpy as np

logger = Logger(__name__)

@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option("--reset", is_flag=True, help="Reset the robot to origin before starting")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option("--parameter_file", default="default_planner.yaml", help="Path to parameter file")
@click.option(
    "--match_method",
    "--match-method",
    type=click.Choice(["class", "feature"]),
    default="feature",
    help="Method to match objects to pick up. Options: class, feature.",
    show_default=True,
)
@click.option(
    "--llm",
    # default="gemma2b",
    default="qwen25-3B-Instruct",
    help="Client to use for language model. Recommended: gemma2b, openai",
    type=click.Choice(get_llm_choices()),
)
@click.option(
    "--realtime",
    "--real-time",
    "--enable-realtime-updates",
    "--enable_realtime_updates",
    is_flag=True,
    help="Enable real-time updates so that the robot will dynamically update its map",
)
@click.option(
    "--device_id",
    default=0,
    help="ID of the device to use for perception",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Set to print debug information",
)
@click.option(
    "--show_intermediate_maps",
    "--show-intermediate-maps",
    is_flag=True,
    help="Set to visualize intermediate maps",
)
@click.option(
    "--target_object",
    "--target-object",
    default="",
    help="Name of the object to pick up",
)
@click.option(
    "--receptacle",
    default="",
    help="Name of the receptacle to place the object in",
)
@click.option(
    "--input-path",
    "-i",
    "--input_file",
    "--input-file",
    "--input",
    "--input_path",
    type=click.Path(),
    default="",
    help="Path to a saved datafile from a previous exploration of the world.",
)
@click.option(
    "--use_llm",
    "--use-llm",
    is_flag=True,
    help="Set to use the language model",
)
@click.option(
    "--use_voice",
    "--use-voice",
    is_flag=True,
    help="Set to use voice input",
)
@click.option(
    "--radius",
    default=3.0,
    type=float,
    help="Radius of the circle around initial position where the robot is allowed to go.",
)
@click.option("--open_loop", "--open-loop", is_flag=True, help="Use open loop grasping")
@click.option(
    "--debug_llm",
    "--debug-llm",
    is_flag=True,
    help="Set to print LLM responses to the console, to debug issues when parsing them when trying new LLMs.",
)



def main(
    robot_ip: str = "192.168.1.15",
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    device_id: int = 0,
    verbose: bool = False,
    show_intermediate_maps: bool = False,
    reset: bool = False,
    target_object: str = "",
    receptacle: str = "",
    match_method: str = "feature",
    llm: str = "gemma",
    use_llm: bool = False,
    use_voice: bool = False,
    open_loop: bool = False,
    debug_llm: bool = False,
    realtime: bool = False,
    radius: float = 3.0,
    input_path: str = "",
):
    """Set up the robot, create a task plan, and execute it."""
    # Get Parameters
    parameters = get_parameters(parameter_file)

    # Create robot
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
    )


    robot.move_to_manip_posture
    obs = robot.get_observation()
    joint_state = obs.joint
    model = robot.get_robot_model()

    pitch_from_vertical = 0.5

    joint_state[HelloStretchIdx.LIFT] = -0.3

    # Strip out fields from the full robot state to only get the 6dof manipulator state
    # TODO: we should probably handle this in the zmq wrapper.
    # arm_cmd = conversions.config_to_manip_command(joint_state)
    robot.switch_to_manipulation_mode()
    robot.arm_to(joint_state, blocking=True)







    # # Create prediction model
    # semantic_sensor = create_semantic_sensor(
    #     parameters=parameters,
    #     device_id=device_id,
    #     verbose=verbose,
    # )

    # # Agents wrap the robot high level planning interface for now
    # agent = RobotAgent(robot, parameters, semantic_sensor, enable_realtime_updates=realtime)
    # print("Starting robot agent: initializing...")
    # agent.start(visualize_map_at_start=show_intermediate_maps)

    # testing_perception = TestPerception("go to navigation mode", agent, retry_on_failure=True)
    # if (testing_perception.can_start()):
    #     testing_perception.run()

    # if (testing_perception.was_successful()):
    #     print("Success")

    # exit_Text = input(colored("You: ", "green"))
    # # Parse things and listen to the user
    # ok = True
    # # while robot.running and ok:
    # #     ok = True

    robot.stop()


class TestPerception(ManagedOperation):

    def can_start(self) -> bool:
        self.attempt("will switch to navigation mode.")
        return True

    def run(self) -> None:
        """Search for a receptacle on the floor."""

        # Update world map
        self.intro("Searching for a receptacle on the floor.")
        self.set_status(status.RUNNING)

        # Must move to nav before we can do anything
        self.robot.move_to_nav_posture()
        # Now update the world
        self.update()

        print(f"So far we have found: {len(self.agent.get_voxel_map().instances)} objects.")

    def was_successful(self) -> bool:
        res = self.robot.in_navigation_mode()
        if res:
            self.cheer("Robot is in navigation mode.")
        else:
            self.error("Robot is still not in navigation mode.")
        return res


if __name__ == "__main__":
    main()
