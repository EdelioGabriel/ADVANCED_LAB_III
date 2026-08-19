"""
subplots.py

Gera painéis de subplots empilhados (estilo científico, minimalista e
legível) a partir de arquivos de dados experimentais tabulares, usando
Matplotlib.

Uso básico (processa todos os arquivos .txt da pasta 'dados'):
    python subplots.py

Um único arquivo, com painel de zoom:
    python subplots.py --arquivo "meu_arquivo.txt" --zoom \
        --zoom-inicio 9.6 --zoom-fim 10.8

Uso programático:
    from subplots import CarregadorDados, PainelEmpilhado, Serie

    carregador = CarregadorDados("dados", delimitador=";")
    df = carregador.carregar("arquivo.txt")

    painel = PainelEmpilhado(
        coluna_x="Corrected time (s)",
        rotulo_x="Tempo (s)",
        series=[
            Serie("WE(1).Potential (V)", "Potencial (V)", cor="tab:blue"),
            Serie("WE(1).Charge (C)", "Carga (C)", cor="tab:red"),
        ],
    )
    fig, eixos = painel.gerar(df, titulo="Meu experimento")
    fig.show()
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter


# ---------------------------------------------------------------------------
# Carregamento de dados
# ---------------------------------------------------------------------------

class CarregadorDados:
    """Carrega arquivos tabulares (.txt, .csv, etc.) de uma pasta em DataFrames."""

    def __init__(self, pasta: str | Path, extensao: str = ".txt", delimitador: str = ";"):
        self.pasta = Path(pasta)
        self.extensao = extensao
        self.delimitador = delimitador

    def listar_arquivos(self) -> list[str]:
        if not self.pasta.is_dir():
            raise FileNotFoundError(f"Pasta não encontrada: {self.pasta}")
        return sorted(p.name for p in self.pasta.iterdir() if p.suffix == self.extensao)

    def carregar(self, nome_arquivo: str) -> pd.DataFrame:
        caminho = self.pasta / nome_arquivo
        return pd.read_csv(caminho, delimiter=self.delimitador)

    def carregar_todos(self) -> dict[str, pd.DataFrame]:
        return {nome: self.carregar(nome) for nome in self.listar_arquivos()}

    def listar_colunas(self, nome_arquivo: str) -> list[str]:
        """Retorna a lista de colunas disponíveis em um arquivo."""
        df = self.carregar(nome_arquivo)
        return list(df.columns)


# ---------------------------------------------------------------------------
# Definição de uma série (uma "caixa" do painel empilhado)
# ---------------------------------------------------------------------------

@dataclass
class Serie:
    """Uma série de dados a ser plotada: coluna Y, rótulo e cor."""

    coluna_y: str
    rotulo: str
    cor: str = "tab:blue"
    # ex: "V", "A", "C" — usada pelo EngFormatter (µ, m, k, ...)
    unidade: str = ""


# ---------------------------------------------------------------------------
# Estilo científico padronizado
# ---------------------------------------------------------------------------

@dataclass
class EstiloCientifico:
    """Parâmetros visuais reutilizáveis para um layout minimalista e legível."""

    fonte: str = "Arial"
    tamanho_fonte: int = 12
    tamanho_titulo: int = 14
    # 0 = sem marcador, só linha (estilo Advanced Materials)
    tamanho_marcador: float = 0.0
    espessura_linha: float = 1.8
    cor_linha_eixo: str = "black"
    cor_grade: str = "lightgray"
    espessura_grade: float = 0.5
    espessura_eixo: float = 1.3
    tamanho_tick: float = 5.0
    largura_fig: float = 9.0
    altura_por_linha: float = 2.3
    altura_1_linha: float = 6
    usar_grade: bool = False       # a maioria dos gráficos da Adv. Mater. não usa grade
    negrito_rotulos: bool = True
    # True se o símbolo µ não renderizar bem com a fonte escolhida
    usar_mathtext: bool = False

    def aplicar_eixo(self, ax, unidade: str = "") -> None:
        """Aplica o estilo (spines, ticks, grade) a um único eixo."""
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(self.cor_linha_eixo)
            spine.set_linewidth(self.espessura_eixo)
        ax.tick_params(
            direction="in", length=self.tamanho_tick, width=self.espessura_eixo,
            colors=self.cor_linha_eixo, labelsize=self.tamanho_fonte,
            top=True, right=True,  # ticks nos 4 lados ("caixa fechada")
        )
        if self.usar_grade:
            ax.grid(True, color=self.cor_grade, linewidth=self.espessura_grade)
        else:
            ax.grid(False)
        ax.set_facecolor("white")
        if unidade:
            ax.yaxis.set_major_formatter(
                EngFormatter(unit=unidade, useMathText=self.usar_mathtext)
            )

    def aplicar_figura(self, fig: Figure, titulo: str = "") -> None:
        """Aplica configurações globais (fundo, fonte, título) à figura."""
        fig.patch.set_facecolor("white")
        # usa a fonte pedida se disponível no sistema; caso contrário,
        # cai para as sans-serif padrão do Matplotlib sem gerar erro
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [self.fonte,
                                           "DejaVu Sans", "Helvetica", "sans-serif"]
        peso = "bold" if self.negrito_rotulos else "normal"
        plt.rcParams["axes.labelweight"] = peso
        plt.rcParams["axes.titleweight"] = peso
        if titulo:
            fig.suptitle(titulo, fontsize=self.tamanho_titulo, y=0.995)


# ---------------------------------------------------------------------------
# Painel de subplots empilhados (com coluna opcional de zoom)
# ---------------------------------------------------------------------------

class PainelEmpilhado:
    """
    Gera um painel de subplots empilhados verticalmente para um conjunto
    arbitrário de séries (qualquer número de colunas Y), com uma coluna
    opcional de "zoom" sobre uma janela do eixo X.
    """

    def __init__(
        self,
        coluna_x: str,
        series: Sequence[Serie],
        rotulo_x: str = "",
        estilo: Optional[EstiloCientifico] = None,
    ):
        if not series:
            raise ValueError(
                "É necessário fornecer ao menos uma série para plotar.")
        self.coluna_x = coluna_x
        self.series = list(series)
        self.rotulo_x = rotulo_x
        self.estilo = estilo or EstiloCientifico()

    def gerar(
        self,
        df: pd.DataFrame,
        titulo: str = "",
        zoom: bool = False,
        janela_zoom: Optional[tuple[float, float]] = None,
    ) -> tuple[Figure, "list[list]"]:
        if zoom and janela_zoom is None:
            raise ValueError(
                "Defina 'janela_zoom' (inicio, fim) para usar zoom=True.")

        # Validação das colunas
        self._validar_colunas(df)

        n = len(self.series)
        ncols = 2 if zoom else 1
        x = df[self.coluna_x]

        fig, eixos = plt.subplots(
            nrows=n, ncols=ncols, sharex="col", squeeze=False,
            figsize=(self.estilo.largura_fig,
                     self.estilo.altura_por_linha * n if n > 1 else self.estilo.altura_1_linha),
        )

        if zoom:
            mascara = (x >= janela_zoom[0]) & (x <= janela_zoom[1])

        marcador = "o" if self.estilo.tamanho_marcador > 0 else None

        for i, serie in enumerate(self.series):
            # sem multiplicação manual — o EngFormatter escala o eixo
            y = df[serie.coluna_y]

            ax_principal = eixos[i][0]
            ax_principal.plot(
                x, y, color=serie.cor, marker=marcador,
                markersize=self.estilo.tamanho_marcador, linewidth=self.estilo.espessura_linha,
            )
            ax_principal.set_ylabel(
                serie.rotulo, fontsize=self.estilo.tamanho_fonte)
            self.estilo.aplicar_eixo(ax_principal, unidade=serie.unidade)

            if zoom:
                ax_zoom = eixos[i][1]
                ax_zoom.plot(
                    x[mascara], y[mascara], color=serie.cor, marker=marcador,
                    markersize=self.estilo.tamanho_marcador, linewidth=self.estilo.espessura_linha,
                )
                self.estilo.aplicar_eixo(ax_zoom, unidade=serie.unidade)

            # esconde rótulos do eixo X em todas as linhas, exceto a última
            if i < n - 1:
                for col in range(ncols):
                    eixos[i][col].tick_params(labelbottom=False)

        eixos[-1][0].set_xlabel(self.rotulo_x,
                                fontsize=self.estilo.tamanho_fonte)
        if zoom:
            eixos[-1][1].set_xlabel(f"{self.rotulo_x} (zoom)",
                                    fontsize=self.estilo.tamanho_fonte)

        self.estilo.aplicar_figura(fig, titulo=titulo)
        fig.tight_layout(rect=(0, 0, 1, 0.97) if titulo else None)
        return fig, eixos

    def _validar_colunas(self, df: pd.DataFrame) -> None:
        """Valida se todas as colunas especificadas existem no DataFrame."""
        colunas_disponiveis = set(df.columns)
        colunas_necessarias = {self.coluna_x}
        for serie in self.series:
            colunas_necessarias.add(serie.coluna_y)

        colunas_faltantes = colunas_necessarias - colunas_disponiveis
        if colunas_faltantes:
            msg = f"Colunas não encontradas: {', '.join(sorted(colunas_faltantes))}\n"
            msg += f"Colunas disponíveis:\n"
            for i, col in enumerate(sorted(colunas_disponiveis), 1):
                msg += f"  {i}. {col}\n"
            raise ValueError(msg)

    @staticmethod
    def salvar(fig: Figure, caminho: str | Path, dpi: int = 300) -> Path:
        caminho = Path(caminho)
        fig.savefig(caminho, dpi=dpi, bbox_inches="tight")
        return caminho


# ---------------------------------------------------------------------------
# Configuração padrão (equivalente às colunas usadas no notebook original)
# ---------------------------------------------------------------------------

def series_padrao() -> list[Serie]:
    """Séries padrão do experimento de cronoamperometria original."""
    return [
        Serie(coluna_y="WE(1).Potential (V)",
              rotulo="Potencial", cor="tab:blue", unidade="V"),
        Serie(coluna_y="WE(1).Charge (C)",
              rotulo="Carga", cor="tab:red", unidade="C"),
        Serie(coluna_y="WE(1).Current (A)", rotulo="Corrente",
              cor="tab:green", unidade="A"),
    ]


# ---------------------------------------------------------------------------
# Interface de linha de comando
# ---------------------------------------------------------------------------

def analisar_argumentos(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera painéis de subplots científicos a partir de arquivos de dados experimentais."
    )
    parser.add_argument("--pasta", default=".",
                        help="Pasta com os arquivos de dados (padrão: pasta atual)")
    parser.add_argument("--extensao", default=".txt",
                        help="Extensão dos arquivos de dados (padrão: .txt)")
    parser.add_argument("--delimitador", default=";",
                        help="Delimitador do arquivo tabular (padrão: ';')")
    parser.add_argument(
        "--arquivo", default=None,
        help="Nome de um arquivo específico a plotar. Se omitido, processa todos os arquivos da pasta.",
    )
    parser.add_argument(
        "--listar-colunas", action="store_true",
        help="Lista todas as colunas disponíveis no primeiro arquivo e sai (útil para descobrir os nomes)."
    )
    parser.add_argument("--coluna-x", default="Corrected time (s)",
                        help="Nome da coluna usada como eixo X")
    parser.add_argument("--rotulo-x", default="Tempo (s)",
                        help="Rótulo do eixo X")
    parser.add_argument(
        "--colunas-y", default=None,
        help="Lista separada por vírgula de colunas Y a plotar (sobrepõe as séries padrão).",
    )
    parser.add_argument(
        "--rotulos-y", default=None,
        help="Lista separada por vírgula de rótulos correspondentes a --colunas-y.",
    )
    parser.add_argument("--zoom", action="store_true",
                        help="Adiciona uma coluna extra com zoom sobre uma janela do eixo X")
    parser.add_argument("--zoom-inicio", type=float,
                        default=None, help="Início da janela de zoom")
    parser.add_argument("--zoom-fim", type=float,
                        default=None, help="Fim da janela de zoom")
    parser.add_argument("--sem-salvar", action="store_true",
                        help="Não salva a imagem em disco")
    parser.add_argument("--sem-exibir", action="store_true",
                        help="Não abre a janela interativa (útil em lote)")
    parser.add_argument("--saida", default=".",
                        help="Pasta de saída para as imagens salvas")
    return parser.parse_args(argv)


def construir_series(args: argparse.Namespace) -> list[Serie]:
    if not args.colunas_y:
        return series_padrao()

    colunas = [c.strip() for c in args.colunas_y.split(",")]
    rotulos = [r.strip() for r in args.rotulos_y.split(",")
               ] if args.rotulos_y else colunas
    if len(rotulos) != len(colunas):
        raise SystemExit(
            "--colunas-y e --rotulos-y devem ter a mesma quantidade de itens.")

    cores_padrao = ["tab:blue", "tab:red", "tab:green",
                    "tab:purple", "tab:orange", "tab:brown"]
    return [
        Serie(coluna_y=col, rotulo=rot,
              cor=cores_padrao[i % len(cores_padrao)])
        for i, (col, rot) in enumerate(zip(colunas, rotulos))
    ]


def exibir_colunas_disponiveis(carregador: CarregadorDados) -> None:
    """Exibe todas as colunas disponíveis do primeiro arquivo."""
    arquivos = carregador.listar_arquivos()
    if not arquivos:
        print(
            f"Nenhum arquivo '{carregador.extensao}' encontrado em '{carregador.pasta}'.")
        return

    primeiro_arquivo = arquivos[0]
    colunas = carregador.listar_colunas(primeiro_arquivo)
    print(f"\nColunas disponíveis em '{primeiro_arquivo}':\n")
    for i, col in enumerate(colunas, 1):
        print(f"  {i:2d}. {col}")
    print(f"\nTotal: {len(colunas)} colunas\n")
    print("Exemplo de uso:")
    print(
        f'  python CA.py --coluna-x "{colunas[0]}" --colunas-y "{colunas[1]},{colunas[2]}" --rotulos-y "Label1,Label2"\n')


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = analisar_argumentos(argv)

    carregador = CarregadorDados(
        args.pasta, extensao=args.extensao, delimitador=args.delimitador)

    # Se --listar-colunas foi especificado, exibe as colunas e sai
    if args.listar_colunas:
        exibir_colunas_disponiveis(carregador)
        return

    if args.zoom and (args.zoom_inicio is None or args.zoom_fim is None):
        raise SystemExit("Use --zoom-inicio e --zoom-fim junto com --zoom.")
    janela_zoom = (args.zoom_inicio, args.zoom_fim) if args.zoom else None

    arquivos = [args.arquivo] if args.arquivo else carregador.listar_arquivos()

    if not arquivos:
        print(
            f"Nenhum arquivo '{args.extensao}' encontrado em '{args.pasta}'.")
        return

    painel = PainelEmpilhado(
        coluna_x=args.coluna_x,
        rotulo_x=args.rotulo_x,
        series=construir_series(args),
    )

    pasta_saida = Path(args.saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    for nome_arquivo in arquivos:
        try:
            df = carregador.carregar(nome_arquivo)
            fig, _ = painel.gerar(
                df,
                titulo=f"Subplots dos dados experimentais — {nome_arquivo}",
                zoom=args.zoom,
                janela_zoom=janela_zoom,
            )

            if not args.sem_salvar:
                caminho = pasta_saida / f"grafico_{nome_arquivo}.png"
                PainelEmpilhado.salvar(fig, caminho)
                print(f"Gráfico salvo como '{caminho}'")

            if not args.sem_exibir:
                plt.show()

            plt.close(fig)
        except ValueError as e:
            print(f"\nErro ao processar '{nome_arquivo}':")
            print(f"{e}")
            print(
                f"\nDica: Use --listar-colunas para descobrir os nomes exatos das colunas.\n")
            continue


if __name__ == "__main__":
    main()
