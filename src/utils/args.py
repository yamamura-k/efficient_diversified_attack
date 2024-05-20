from argparse import ArgumentParser


def argparser():
    parser = ArgumentParser()

    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory name (not path)",
        required=True,
        type=str,
    )
    parser.add_argument("-p", "--param", required=True, nargs="*", type=str)
    parser.add_argument(
        "--cmd_param",
        type=str,
        default=None,
        nargs="*",
        help='list of "param:cast type:value"' + " ex)  model_name:str:XXX solver.params.num_sample:int:10",
    )
    parser.add_argument("-g", "--gpu", type=int, default=None)
    parser.add_argument("-bs", "--batch_size", type=int, default=None)
    parser.add_argument("--n_threads", type=int, default=1)
    parser.add_argument(
        "--image_indices",
        type=str,
        default=None,
        help="path to yaml file which contains target image indices",
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=30,
        help="10:DEBUG,20:INFO,30:WARNING,40:ERROR,50:CRITICAL",
    )
    parser.add_argument("--export_level", type=int, default=60, choices=[10, 20, 30, 40, 50, 60])
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="attack all images when this flag is on",
    )
    return parser
