import os


def generate_scp_files(mix_path: str, ref_paths: list, output_dir: str):
    """
    :param mix_path: запись лекции
    :param ref_paths: референсы голоса преподавателя
    :param output_dir: папка сохранение scp файлов
    :return: scp файлы для TSE
    """
    mix_scp = os.path.join(output_dir, "mix.scp")
    aux_scp = os.path.join(output_dir, "aux.scp")

    mix_name = os.path.splitext(os.path.basename(mix_path))[0]

    with open(mix_scp, 'w') as f_mix, open(aux_scp, "w") as f_aux:
        for ref_path in ref_paths:
            ref_name = os.path.splitext(os.path.basename(ref_path))[0]
            utt_id = f'{mix_name}_{ref_name}'

            f_mix.write(f'{utt_id} {os.path.abspath(mix_path)}\n')
            f_aux.write(f'{utt_id} {os.path.abspath(ref_path)}\n')


    return mix_scp, aux_scp