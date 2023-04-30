import os
import discord
from dotenv import load_dotenv
import random
import re

# GENERATE

#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow.compat.v1 as tf

import model, sample, encoder

def sample_model(
    model_name='h5',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=250,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
):
    """
    Run the sample_model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to
     reproduce results
    :nsamples=0 : Number of samples to return, if 0, continues to
     generate samples indefinately.
    :batch_size=1 : Number of batches (only affects speed/memory).
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=length,
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )[:, 1:]

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        generated = 0
        while nsamples == 0 or generated < nsamples:
            out = sess.run(output)
            for i in range(batch_size):
                generated += batch_size
                text = '```'
                text += enc.decode(out[i])
                text += '```'
                #print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
                return text


# RESPOND

def interact_model(
    model_name='h5',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=100,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
    raw_text='test',
    inline = True
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        # Generation Code

        context_tokens = enc.encode(raw_text)
        generated = 0
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = ''
                if inline:
                    text = '```'
                text += enc.decode(out[i])
                if inline:
                    text += '```'
                return text
        print("=" * 80)

# DISCORD BOT
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
TOKEN = ('YOUR TOKEN HERE')


@client.event
async def on_ready():
    print("Logged in as a bot {0.user}".format(client))

@client.event
async def on_message(message):

    if message.author == client.user:
        return

    if message.content == '!g':
        response = sample_model()
        await message.channel.send(response)
    elif "!g" in message.content[:2]:
        text = message.content[3:]
        content = interact_model(raw_text = text, inline = True, length = 250)
        index = content.find('\n')
        await message.channel.send(content[index:])
    elif "!r" in message.content[:2]:
        text = message.content[3:]
        
        content = interact_model(raw_text = text, inline = False, length = 50)
        
        regexp = re.compile('^[A-z]{3} [0-9]{2}:[0-9]{2} [A-Z]{2} - .+: "(.*)"')
        matched = regexp.match(content)
        
        if matched:
            toReturn = matched.group(1)
            await message.channel.send(toReturn)
        else:
            index = (content.find('\n')) + 1
            temp = content[index:]
            
            temp_match = regexp.match(temp)
            if temp_match:
                toReturn = temp_match.group(1)
                await message.channel.send(toReturn)
            else:
                await message.channel.send(temp)
            
    elif "!c" in message.content[:2]:
        # remove anything after \n
        # add text to response
        text = message.content[3:]
        content = interact_model(raw_text = text, length = 20, inline = False)
        
        sep = '\n'
        stripped = content.split(sep, 1)[0]
        
        toReturn = text + "" + stripped
        
        await message.channel.send(toReturn[:(len(toReturn) - 1)])
    elif message.content =='!h':
        response = '```!g <prompt> - Generates Conversation. If no prompt provided a random one will be used.\n!r <prompt> - Responds to prompt\n!c <prompt> - Continues prompt```'
        await message.channel.send(response)


client.run(TOKEN)