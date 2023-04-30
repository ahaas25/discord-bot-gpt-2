## Overview

A very quick and dirty implementation on Open AI's GPT-2 in a Discord Bot. This was hacked together in a day and is by no means the best or most efficient way to do it. I may come back to this project to make it prettier another time.

## Requirements

* General understanding of how to create and manage Discord bots
* A trained GPT model
* Basic understanding of Python
* All the pre-requisites for using GPT 2 which this was forked from
* A CUDA-Capable GPU (Recommended for better performance)

## Usage

You'll have to modify ``bot.py`` with your Discord bot token, as well as point the commands to the correct model. ``bot.py`` is a very basic Discord bot with Open AI's generation scripts pasted into it and modified to return a string rather than print to console. My model was trained off a group chat with friends, as such I've written the bot and its commands to reflect this. The data format which I trained my model on follows the regular expression ``^[A-z]{3} [0-9]{2}:[0-9]{2} [A-Z]{2} - .+: "(.*)"``, which looks like ``Sep 05:57 PM - Username: "Message content"``. Unless your model outputs text in this exact format, you will have to modify this bot to accommodate your needs

There are three commands to interact with this bot. 
* ``!g <prompt>``, which generates based off a prompt, or a random one if none is provided.
* ``!r <prompt>``, which replies to a prompt. Example ``!r Hi, how are you?`` may respond ``I'm good!``. This command assumes the model to return a string following the regexp mentioned above, and cuts out the irrelevant information to mimic a response from the bot.
* ``!c <prompt>``, which continues a prompt. Example ``!c My name is`` to which the bot may continue the prompt with ``My name is Jojgo``

## License

[Modified MIT](./LICENSE)
