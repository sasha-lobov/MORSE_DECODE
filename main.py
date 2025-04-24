import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from aiogram import Bot, Dispatcher, types
from aiogram.types import ParseMode
from aiogram.dispatcher.filters import Command
from aiogram.utils import executor
from dotenv import load_dotenv
import json
from model import MorseNet, create_vocab, predict_single, norm_signal, save_signal


load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.answer("Привет! Отправь мне аудиофайл, и я расшифрую его по азбуке Морзе.")

@dp.message_handler(content_types=types.ContentType.ANY)
async def handle_audio(message: types.Message):
    if not message.audio and not message.voice and not message.document:
        await message.reply("Пожалуйста, пришли аудиофайл.")
        return

    file = message.audio or message.voice or message.document
    path = f"temp/{file.file_id}.opus"

    file_obj = await file.download(destination_file=path)

    try:
        y, sr = librosa.load(path)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, fmax=8000)
        S = S * S
        S_dB = librosa.power_to_db(S, ref=np.max)
        start = int(np.argmax(S_dB) / S_dB.shape[1])
        S_dB = S_dB[start:start + 2]
        signal = norm_signal(S_dB[0])

        temp_path = f"temp/{file.file_id}.json"
        save_signal(signal, temp_path)

        text = predict_single(temp_path, model, idx2char)

        await message.reply(f"📡 Расшифровка:\n<code>{text}</code>", parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.reply(f"Ошибка при обработке: {e}")
    finally:
        os.remove(path)
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    
    char_list = list("АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789# ")
    char2idx, idx2char = create_vocab(char_list)
    num_classes = len(char2idx)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MorseNet(num_classes=num_classes)
    model.load_state_dict(torch.load("cnn_gru.pth", map_location=device))
    model = model.to(device)

    os.makedirs("temp", exist_ok=True)
    executor.start_polling(dp, skip_updates=True)