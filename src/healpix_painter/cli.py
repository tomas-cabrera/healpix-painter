import typer
from healpix_painter import basicpainter

app = typer.Typer()
app.command()(basicpainter)

if __name__ == "__main__":
    app()
