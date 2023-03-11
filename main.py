from src.utils.utils import cleanup, get_args, setup


def main():
    """Main function."""
    args = get_args()
    setup(args)

    # TODO: Add your code here

    cleanup(args)


if __name__ == "__main__":
    main()
