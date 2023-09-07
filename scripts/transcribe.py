import stable_whisper
import torchaudio
import argparse
import os


def split_text_and_write_metadata(
    audio_file_path: str,
    data_list,
    parent_path: str,
    iteration: int,
    char_name: str,
    resample: int,
):
    metadata_file_path = os.path.join(parent_path, "metadata.csv")
    wav_path = os.path.join(parent_path, "wavs")

    if not os.path.exists(wav_path):
        os.makedirs(wav_path)

    with open(metadata_file_path, "a") as metadata_file:
        for i, obj in enumerate(data_list):
            start_time = obj["start"]
            end_time = obj["end"] + 0.3

            duration = end_time - start_time

            text = obj["text"].strip()

            if duration < 0.5:
                print(f"{text} | {duration:.2f}s - Segment is too small, skipping...\n")
                continue
            elif duration > 10.0:
                print(f"{text} | {duration:.2f}s - Segment is too long, skipping...\n")
                continue
            else:
                print(f"{text} | {duration:.2f}s")

            metadata_line = f"{char_name}_{iteration}_{i}|{text}\n"
            metadata_file.write(metadata_line)
            output_file_name = f"{char_name}_{iteration}_{i}.wav"
            # Update the output directory as needed
            output_file_path = os.path.join(wav_path, output_file_name)

            wav, sr = torchaudio.load(  # type: ignore
                audio_file_path,
                frame_offset=0,
                num_frames=-1,
                normalize=True,
                channels_first=True,
            )

            if resample == None:
                target_sr = sr
            else:
                target_sr = resample

            wav = wav.mean(dim=0).unsqueeze(0)

            if sr != target_sr:
                wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(
                    wav
                )

            if wav.shape[1] / target_sr > 10:
                wav = wav[
                    :,
                    int(float(start_time * target_sr)) : int(
                        float(end_time * target_sr)
                    ),
                ]

            torchaudio.save(  # type: ignore
                output_file_path,
                wav,
                target_sr,
                channels_first=True,
                bits_per_sample=16,
                encoding="PCM_S",
            )

    os.remove(audio_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default="./")
    parser.add_argument("--enable-demucs", type=bool, default=False)
    parser.add_argument(
        "-s",
        "--resample",
        type=int,
        default=None,
        help="Specify the final sample rate of the transcribed files",
    )
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Where to download the Whisper models",
    )
    parser.add_argument(
        "--model", "-m", type=str, default="medium", help="Which Whisper model to use"
    )
    args = parser.parse_args()

    parent_dir = args.input_dir

    speaker_names = list(os.walk(parent_dir))[0][1]
    total_files = sum([len(files) for _, _, files in os.walk(parent_dir)])

    model = stable_whisper.load_model(args.model, download_root=args.model_dir)
    # this modified model run just like the original model but accepts additional arguments

    processed_files = 0
    for speaker in speaker_names:
        spk_dir = os.path.join(parent_dir, speaker)

        file_list = len(
            [f for f in os.listdir(spk_dir) if os.path.isfile(os.path.join(spk_dir, f))]
        )

        meta_path = os.path.join(spk_dir, "metadata.csv")

        if os.path.exists(meta_path):
            print(f"Meta file found, skipping {speaker}")
            total_files -= file_list
            continue

        print(f"Transcribing audios from {speaker}")

        for i, wavfile in enumerate(list(os.walk(spk_dir))[0][2]):
            original_mp3 = os.path.join(spk_dir, wavfile)

            print(f"Working on file: {original_mp3}")

            decode_opts: dict[str, str] = dict()
            try:
                result = model.transcribe(
                    original_mp3,
                    vad=False,
                    demucs=args.enable_demucs,
                    language=args.language,
                )

                (
                    result.split_by_punctuation([(".", " "), "。", "?"])  # type: ignore
                    .split_by_gap(0.5)
                    .merge_by_gap(0.15, max_words=3)
                    .merge_by_punctuation([(",", " "), ", "])
                    .split_by_punctuation([(".", " "), "。", "?"])
                    .clamp_max(None, 10)
                )

                flist = result.to_dict()["segments"]  # type: ignore
            except:
                print(f"Failed to transcribe file: {original_mp3}")

            split_text_and_write_metadata(
                original_mp3, flist, spk_dir, i, speaker, args.resample
            )

            processed_files += 1
            print(f"Processed: {processed_files}/{total_files}")
