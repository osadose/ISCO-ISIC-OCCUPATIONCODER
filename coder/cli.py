import click
from .pipeline import run_pipeline

@click.command()
@click.option("--input", "input_csv", required=True, help="Path to input CSV from enumerators")
@click.option("--output", "output_csv", required=True, help="Path to output cleaned CSV")
@click.option("--isco", "isco_xlsx", required=True, help="Path to ISCO Excel file")
@click.option("--isic", "isic_xlsx", required=True, help="Path to ISIC Excel file")
@click.option("--config", "config_path", default=None, help="Optional YAML config path")
def main(input_csv, output_csv, isco_xlsx, isic_xlsx, config_path):
    """
    NBS ISCO & ISIC Cleaning Tool (CLI)
    """
    df, review_path = run_pipeline(
        input_csv=input_csv,
        output_csv=output_csv,
        isco_xlsx=isco_xlsx,
        isic_xlsx=isic_xlsx,
        config_path=config_path
    )
    click.echo(f"Saved cleaned file to: {output_csv}")
    if review_path:
        click.echo(f"Saved review queue to: {review_path}")


if __name__ == "__main__":
    main()