import stable_whisper
import torchaudio
import argparse
import os

target_sr = 16000


def split_text_and_write_metadata(
    audio_file_path: str, data_list, parent_path: str, iteration: int, char_name: str, resample: int
):
    metadata_file_path = parent_path + "/metadata.csv"

    if not os.path.exists(parent_path + "/wavs/"):
        os.makedirs(parent_path + "/wavs/")

    with open(metadata_file_path, "a") as metadata_file:
        for i, obj in enumerate(data_list):
            start_time = obj["start"]
            end_time = obj["end"] + 0.2

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
            output_file_path = parent_path + f"/wavs/{output_file_name}"

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
                wav = wav[:, int(float(start_time * target_sr)) : int(float(end_time * target_sr))]


            
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
    parser.add_argument("--input", default="./")
    parser.add_argument("--enable-demucs", type=bool, default=False)
    parser.add_argument("--resample", type=int, default=None)
    parser.add_argument("--language", type=str, default=None)
    args = parser.parse_args()

    parent_dir = args.input

    speaker_names = list(os.walk(parent_dir))[0][1]
    total_files = sum([len(files) for _, _, files in os.walk(parent_dir)])

    model = stable_whisper.load_model(
        "medium", download_root="/mnt/Mass/Resources/Whisper"
    )
    # this modified model run just like the original model but accepts additional arguments

    processed_files = 0
    for speaker in speaker_names:
        print(f"Transcribing audios from {speaker}")
        spk_dir = parent_dir + "/" + speaker

        if os.path.exists(spk_dir + "/metadata.csv"):
            print(f"Skipping {speaker}")
            continue

        for i, wavfile in enumerate(list(os.walk(spk_dir))[0][2]):
            original_mp3 = parent_dir + "/" + speaker + "/" + wavfile

            print(f"Working on file: {original_mp3}")

            decode_opts: dict[str, str] = dict()

            if args.language != None:
                decode_opts.update({"language": args.language})

            result = model.transcribe(
                original_mp3, vad=False, demucs=args.enable_demucs, decode_options=decode_opts
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

            split_text_and_write_metadata(original_mp3, flist, spk_dir, i, speaker, args.resample)

            processed_files += 1
            print(f"Processed: {processed_files}/{total_files}")
