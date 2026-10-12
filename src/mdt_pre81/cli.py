import argparse
from .pipeline import ETAPAS, executar


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mdt_pre81", description="MDT pré-1981 Buritama/Zacarias + webmap")
    ap.add_argument("etapas", nargs="+", choices=ETAPAS + ["cartas", "all"],
                    help="etapa(s): " + ", ".join(ETAPAS) + " ou 'all'")
    ap.add_argument("-c", "--config", default="config/projeto.yml")
    a = ap.parse_args(argv)
    executar(a.etapas, a.config)


if __name__ == "__main__":
    main()
