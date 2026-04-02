try:
    # Works when launched from repository root.
    from corner.ui import main
except ModuleNotFoundError:
    # Works when this file is used directly as the Streamlit entrypoint.
    from ui import main


if __name__ == "__main__":
    main()
