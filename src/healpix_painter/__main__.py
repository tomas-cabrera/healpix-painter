if __name__ == "__main__":
    import sys

    from healpix_painter.cli import app

    if len(sys.argv) == 1:
        app()
    else:
        app(prog_name=sys.argv[0])
