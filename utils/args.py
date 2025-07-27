from argparse import ArgumentParser, Namespace

def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods

    Args:
        parser: the parser instance

    Returns:
        None
    """
    group = parser.add_argument_group('Rehearsal arguments', 'Arguments shared by all rehearsal-based methods.')

    group.add_argument('--buffer_size', type=int, required=True,
                       help='The size of the memory buffer.')
    group.add_argument('--minibatch_size', type=int,
                       help='The batch size of the memory buffer.')