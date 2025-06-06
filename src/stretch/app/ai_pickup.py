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
from stretch.app.speech_to_text import speech_to_text
from termcolor import colored
import time
import gc
import torch
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os


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


# def get_object_and_receptacle(use_llm, use_voice, prompt, llm):

    

#     return llm_response


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


    # Use LLM for pick and place before creating robot object

    # Prompt defines the type of LLM model
    prompt = PickupPromptBuilder()

    # Getting object and receptacle
    # llm_response = get_object_and_receptacle(use_llm, use_voice, prompt, llm)

    # edge case for voice
    if use_voice and not use_llm:
        logger.warning("Voice input is only supported with a language model.")
        logger.warning(
            "Please set --use-llm to use voice input. For now, we will disable voice input."
        )
        use_voice = False

    # Using llm 
    llm_client = None
    if use_llm:
        if use_voice:
            whisper = speech_to_text()
            audio = whisper.record_audio()
            input_text = whisper.speech_to_text(audio)
        else:
            input_text = input(colored("You: ", "green"))
        llm_client = get_llm_client(llm, prompt=prompt)
        assistant_response = llm_client(input_text)
        llm_response = prompt.parse_response(assistant_response)
    # not using llm
    else:
        if len(target_object) == 0:
            target_object = input("Enter the target object: ")
        if len(receptacle) == 0:
            receptacle = input("Enter the target receptacle: ")
        llm_response = [("pickup", target_object), ("place", receptacle)]

    # debugging
    print(llm_response)

    if llm_client is not None:
        del llm_client
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()  # Clear unused memory
    llm_client = None
    time.sleep(2)


    # Create robot
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    # Create prediction model
    robot.say_sync("Let's start the pick and place demo !")
    time.sleep(0.5)
    robot.say_sync("Initializing sensors.")
    semantic_sensor = create_semantic_sensor(
        parameters=parameters,
        device_id=device_id,
        verbose=verbose,
    )

    # Agents wrap the robot high level planning interface for now
    agent = RobotAgent(robot, parameters, semantic_sensor, enable_realtime_updates=realtime)
    print("Starting robot agent: initializing...")
    agent.start(visualize_map_at_start=show_intermediate_maps)
    if reset:
        print("Reset: moving robot to origin")
        agent.move_closed_loop([0, 0, 0], max_time=60.0)

    if radius is not None and radius > 0:
        print("Setting allowed radius to:", radius)
        agent.set_allowed_radius(radius)

    # Load a PKL file from a previous run and process it
    # This will use ICP to match current observations to the previous ones
    # ANd then update the map with the new observations
    if input_path is not None and len(input_path) > 0:
        print("Loading map from:", input_path)
        agent.load_map(input_path)

    robot.say_sync("Sensors Initialized")

    # Executor handles outputs from the LLM client and converts them into executable actions
    executor = PickupExecutor(
        robot, agent, available_actions=prompt.get_available_actions(), dry_run=False
    )

    # Parse things and listen to the user

    ok = True
    agent.reset()
    say_this = None
    ok = executor(llm_response)


    # Debugging code
    # debug_response = [('say', '"I am picking up the bottle and placing it on the table."'), ('pickup', 'bottle'), ('place', 'table')]
    # debug_response = [('say', '"I am picking up the bottle and giving it to you."'), ('pickup', 'bottle'), ('hand_over', '')]
    # debug_response = [('say', '"I am picking up the bottle and giving it to you."'), ('pickup', 'bottle')]

    # ok = executor(debug_response)

    robot.stop()
    os.system("pkill rerun") 


if __name__ == "__main__":
    main()
