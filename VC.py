"""
VC.py

Gera gráficos de Voltametria Cíclica (estilo científico, minimalista e
legível) a partir de arquivos de dados experimentais tabulares, usando
Matplotlib. Permite colorir os ciclos de varredura (Scan) e filtrar os
dados para plotar a partir de (ou exatamente) um determinado ciclo.

Autor: Edélio Gabriel M. de Jesus

Uso básico (processa todos os arquivos .txt da pasta 'dados'):
    python VC.py

Um único arquivo, plotando a partir do 3º ciclo em diante:
    python VC.py --arquivo "meu_arquivo.txt" --ciclo-inicial 3

Um único ciclo específico (ex: só o ciclo 5):
    python VC.py --arquivo "meu_arquivo.txt" --ciclo-inicial 5 --ciclo-exato

Sobrepor as curvas de dois arquivos no mesmo gráfico:
    python VC.py --arquivo "arquivo1.txt" "arquivo2.txt" --comparativo

Uso programático:
    from VC import CarregadorDados, PainelVoltametria

    carregador = CarregadorDados("dados", delimitador=";")
    df = carregador.carregar("arquivo.txt")

    painel = PainelVoltametria(
        coluna_x="WE(1).Potential (V)",
        coluna_y="WE(1).Current (A)",
        coluna_ciclo="Scan",
    )
    fig, ax = painel.gerar(df, titulo="Meu experimento", ciclo_inicial=3)
    fig.show()
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
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
# Estilo científico padronizado (mesmos parâmetros usados em CA.py)
# ---------------------------------------------------------------------------

@dataclass
class EstiloCientifico:
    """Parâmetros visuais reutilizáveis para um layout minimalista e legível."""

    fonte: str = "Arial"
    tamanho_fonte: int = 12
    tamanho_titulo: int = 14
    espessura_linha: float = 1.3
    cor_linha_eixo: str = "black"
    cor_grade: str = "lightgray"
    espessura_grade: float = 0.5
    espessura_eixo: float = 1.3
    tamanho_tick: float = 5.0
    largura_fig: float = 7.0
    altura_fig: float = 6.0
    usar_grade: bool = False       # a maioria dos gráficos da Adv. Mater. não usa grade
    negrito_rotulos: bool = True
    # True se o símbolo µ não renderizar bem com a fonte escolhida
    usar_mathtext: bool = False

    def aplicar_eixo(self, ax: Axes, unidade_x: str = "", unidade_y: str = "") -> None:
        """Aplica o estilo (spines, ticks, grade, formatação dos eixos) a um Axes."""
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
        if unidade_x:
            ax.xaxis.set_major_formatter(
                EngFormatter(unit=unidade_x, useMathText=self.usar_mathtext)
            )
        if unidade_y:
            ax.yaxis.set_major_formatter(
                EngFormatter(unit=unidade_y, useMathText=self.usar_mathtext)
            )

    def aplicar_figura(self, fig: Figure, titulo: str = "") -> None:
        """Aplica configurações globais (fundo, fonte, título) à figura."""
        fig.patch.set_facecolor("white")
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [self.fonte,
                                           "DejaVu Sans", "Helvetica", "sans-serif"]
        peso = "bold" if self.negrito_rotulos else "normal"
        plt.rcParams["axes.labelweight"] = peso
        plt.rcParams["axes.titleweight"] = peso
        if titulo:
            fig.suptitle(titulo, fontsize=self.tamanho_titulo, y=0.98)


# ---------------------------------------------------------------------------
# Painel de Voltametria Cíclica
# ---------------------------------------------------------------------------

class PainelVoltametria:
    """
    Gera o gráfico de Voltametria Cíclica (Corrente x Potencial), com
    coloração opcional por ciclo de varredura e filtro para selecionar a
    partir de qual ciclo (ou exatamente qual ciclo) plotar.
    """

    def __init__(
        self,
        coluna_x: str,
        coluna_y: str,
        coluna_ciclo: str = "Scan",
        rotulo_x: str = "Potencial (V)",
        rotulo_y: str = "Corrente",
        unidade_x: str = "",
        unidade_y: str = "A",
        colormap: str = "plasma",
        estilo: Optional[EstiloCientifico] = None,
    ):
        self.coluna_x = coluna_x
        self.coluna_y = coluna_y
        self.coluna_ciclo = coluna_ciclo
        self.rotulo_x = rotulo_x
        self.rotulo_y = rotulo_y
        self.unidade_x = unidade_x
        self.unidade_y = unidade_y
        self.colormap = colormap
        self.estilo = estilo or EstiloCientifico()

    def gerar(
        self,
        df: pd.DataFrame,
        titulo: str = "",
        ciclo_inicial: Optional[int] = None,
        ciclo_exato: bool = False,
        cor_por_ciclo: bool = True,
    ) -> tuple[Figure, Axes]:
        if ciclo_exato and ciclo_inicial is None:
            raise ValueError(
                "Defina 'ciclo_inicial' para usar ciclo_exato=True.")

        tem_coluna_ciclo = self.coluna_ciclo in df.columns

        # Sem coluna de ciclo: não há como colorir por ciclo nem filtrar.
        # Se o usuário pediu explicitamente um filtro, isso é um erro real;
        # caso contrário, apenas cai para uma curva única, sem quebrar.
        if not tem_coluna_ciclo:
            if ciclo_inicial is not None:
                raise ValueError(
                    f"Coluna de ciclo '{self.coluna_ciclo}' não encontrada neste "
                    f"arquivo — não é possível aplicar o filtro de ciclo. "
                    f"Use --coluna-ciclo para indicar a coluna correta, ou remova "
                    f"o filtro de ciclo."
                )
            cor_por_ciclo = False

        self._validar_colunas(df, exigir_coluna_ciclo=tem_coluna_ciclo)
        df_plot = self._filtrar_ciclos(
            df, ciclo_inicial, ciclo_exato, tem_coluna_ciclo=tem_coluna_ciclo)

        fig, ax = plt.subplots(
            figsize=(self.estilo.largura_fig, self.estilo.altura_fig))

        if cor_por_ciclo and tem_coluna_ciclo:
            ciclos = np.sort(df_plot[self.coluna_ciclo].unique())
            cores = plt.get_cmap(self.colormap)(np.linspace(0, 1, len(ciclos)))
            for ciclo, cor in zip(ciclos, cores):
                dados_ciclo = df_plot[df_plot[self.coluna_ciclo] == ciclo]
                ax.plot(
                    dados_ciclo[self.coluna_x], dados_ciclo[self.coluna_y],
                    color=cor, linewidth=self.estilo.espessura_linha,
                    label=f"Ciclo {ciclo}",
                )
            ax.legend(fontsize=self.estilo.tamanho_fonte - 2,
                      frameon=False, loc="best")
        else:
            ax.plot(
                df_plot[self.coluna_x], df_plot[self.coluna_y],
                color="tab:blue", linewidth=self.estilo.espessura_linha,
            )

        ax.set_xlabel(self.rotulo_x, fontsize=self.estilo.tamanho_fonte)
        ax.set_ylabel(self.rotulo_y, fontsize=self.estilo.tamanho_fonte)
        self.estilo.aplicar_eixo(
            ax, unidade_x=self.unidade_x, unidade_y=self.unidade_y)
        self.estilo.aplicar_figura(fig, titulo=titulo)
        fig.tight_layout(rect=(0, 0, 1, 0.97) if titulo else None)
        return fig, ax

    def _filtrar_ciclos(
        self,
        df: pd.DataFrame,
        ciclo_inicial: Optional[int],
        ciclo_exato: bool,
        tem_coluna_ciclo: bool = True,
    ) -> pd.DataFrame:
        """Aplica o filtro de ciclos: a partir de 'ciclo_inicial', ou
        exatamente nele quando ciclo_exato=True. Sem coluna de ciclo,
        não há filtro possível — retorna os dados como estão."""
        if tem_coluna_ciclo and ciclo_inicial is not None:
            if ciclo_exato:
                df_plot = df[df[self.coluna_ciclo] == ciclo_inicial]
            else:
                df_plot = df[df[self.coluna_ciclo] >= ciclo_inicial]
        else:
            df_plot = df.copy()

        if df_plot.empty:
            raise ValueError(
                "Nenhum dado para plotar após aplicar o filtro de ciclos.")

        colunas_para_dropna = [self.coluna_x, self.coluna_y]
        if tem_coluna_ciclo:
            colunas_para_dropna.append(self.coluna_ciclo)
        return df_plot.dropna(subset=colunas_para_dropna)

    def _validar_colunas(self, df: pd.DataFrame, exigir_coluna_ciclo: bool = True) -> None:
        """Valida se todas as colunas especificadas existem no DataFrame."""
        colunas_disponiveis = set(df.columns)
        colunas_necessarias = {self.coluna_x, self.coluna_y}
        if exigir_coluna_ciclo:
            colunas_necessarias.add(self.coluna_ciclo)
        colunas_faltantes = colunas_necessarias - colunas_disponiveis
        if colunas_faltantes:
            msg = f"Colunas não encontradas: {', '.join(sorted(colunas_faltantes))}\n"
            msg += f"Colunas disponíveis:\n"
            for i, col in enumerate(sorted(colunas_disponiveis), 1):
                msg += f"  {i}. {col}\n"
            raise ValueError(msg)

    def gerar_comparativo(
        self,
        dados: dict[str, pd.DataFrame],
        titulo: str = "",
        ciclo_inicial: Optional[int] = None,
        ciclo_exato: bool = False,
    ) -> tuple[Figure, Axes]:
        """Sobrepõe a curva de Voltametria Cíclica de vários arquivos no
        mesmo gráfico — uma cor por arquivo (o filtro de ciclo, se dado,
        é aplicado igualmente a todos os arquivos)."""
        if not dados:
            raise ValueError(
                "É necessário fornecer ao menos um arquivo para comparar.")
        if ciclo_exato and ciclo_inicial is None:
            raise ValueError(
                "Defina 'ciclo_inicial' para usar ciclo_exato=True.")

        fig, ax = plt.subplots(
            figsize=(self.estilo.largura_fig, self.estilo.altura_fig))
        cores = plt.get_cmap("tab10")

        for indice, (nome_arquivo, df) in enumerate(dados.items()):
            tem_coluna_ciclo = self.coluna_ciclo in df.columns
            if not tem_coluna_ciclo and ciclo_inicial is not None:
                raise ValueError(
                    f"Coluna de ciclo '{self.coluna_ciclo}' não encontrada em "
                    f"'{nome_arquivo}' — não é possível aplicar o filtro de ciclo."
                )

            self._validar_colunas(df, exigir_coluna_ciclo=tem_coluna_ciclo)
            df_plot = self._filtrar_ciclos(
                df, ciclo_inicial, ciclo_exato, tem_coluna_ciclo=tem_coluna_ciclo)

            ax.plot(
                df_plot[self.coluna_x], df_plot[self.coluna_y],
                color=cores(indice % 10), linewidth=self.estilo.espessura_linha,
                label=Path(nome_arquivo).stem,
            )

        ax.set_xlabel(self.rotulo_x, fontsize=self.estilo.tamanho_fonte)
        ax.set_ylabel(self.rotulo_y, fontsize=self.estilo.tamanho_fonte)
        self.estilo.aplicar_eixo(
            ax, unidade_x=self.unidade_x, unidade_y=self.unidade_y)
        ax.legend(fontsize=self.estilo.tamanho_fonte - 2, frameon=False, loc="best")
        self.estilo.aplicar_figura(fig, titulo=titulo)
        fig.tight_layout(rect=(0, 0, 1, 0.97) if titulo else None)
        return fig, ax

    @staticmethod
    def salvar(fig: Figure, caminho: str | Path, dpi: int = 300) -> Path:
        caminho = Path(caminho)
        fig.savefig(caminho, dpi=dpi, bbox_inches="tight")
        return caminho


# ---------------------------------------------------------------------------
# Configuração padrão (equivalente às colunas usadas no notebook original)
# ---------------------------------------------------------------------------

def config_padrao() -> dict:
    """Parâmetros padrão do experimento de voltametria cíclica original."""
    return dict(
        coluna_x="WE(1).Potential (V)",
        coluna_y="WE(1).Current (A)",
        coluna_ciclo="Scan",
        rotulo_x="Potencial (V) - Ag/AgCl (KCl)",
        rotulo_y="Corrente",
    )


# ---------------------------------------------------------------------------
# Interface de linha de comando
# ---------------------------------------------------------------------------

def analisar_argumentos(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera gráficos de Voltametria Cíclica a partir de arquivos de dados experimentais."
    )
    parser.add_argument("--pasta", default=".",
                        help="Pasta com os arquivos de dados (padrão: pasta atual)")
    parser.add_argument("--extensao", default=".txt",
                        help="Extensão dos arquivos de dados (padrão: .txt)")
    parser.add_argument("--delimitador", default=";",
                        help="Delimitador do arquivo tabular (padrão: ';')")
    parser.add_argument(
        "--arquivo", nargs="+", default=None,
        help="Um ou mais arquivos específicos a plotar (separados por espaço). "
             "Se omitido, processa todos os arquivos da pasta. Combine com "
             "--comparativo para sobrepor as curvas selecionadas no mesmo gráfico.",
    )
    parser.add_argument(
        "--listar-colunas", action="store_true",
        help="Lista todas as colunas disponíveis no primeiro arquivo e sai (útil para descobrir os nomes)."
    )

    parser.add_argument("--coluna-x", default="WE(1).Potential (V)",
                        help="Nome da coluna usada como eixo X (potencial)")
    parser.add_argument("--coluna-y", default="WE(1).Current (A)",
                        help="Nome da coluna usada como eixo Y (corrente)")
    parser.add_argument("--coluna-ciclo", default="Scan",
                        help="Nome da coluna que identifica o número do ciclo/varredura")
    parser.add_argument("--rotulo-x", default="Potencial (V)",
                        help="Rótulo do eixo X")
    parser.add_argument("--rotulo-y", default="Corrente",
                        help="Rótulo do eixo Y")
    parser.add_argument("--titulo", default=None,
                        help="Título do gráfico (padrão: 'Voltametria Cíclica — <arquivo>')")

    # Argumentos específicos da técnica de VC
    parser.add_argument(
        "--ciclo-inicial", type=int, default=None,
        help="A partir de qual ciclo plotar (inclusive). Combine com --ciclo-exato "
             "para plotar SOMENTE esse ciclo.",
    )
    parser.add_argument(
        "--ciclo-exato", action="store_true",
        help="Plota apenas o ciclo indicado em --ciclo-inicial, em vez de todos a partir dele.",
    )
    parser.add_argument(
        "--sem-cor-por-ciclo", action="store_true",
        help="Plota todos os ciclos filtrados como uma única curva, sem colorir por ciclo.",
    )
    parser.add_argument("--colormap", default="plasma",
                        help="Colormap do Matplotlib usado para colorir os ciclos (padrão: 'plasma')")
    parser.add_argument(
        "--comparativo", action="store_true",
        help="Sobrepõe as curvas de todos os arquivos selecionados (--arquivo) "
             "em um único gráfico, uma cor por arquivo.",
    )

    parser.add_argument("--sem-salvar", action="store_true",
                        help="Não salva a imagem em disco")
    parser.add_argument("--sem-exibir", action="store_true",
                        help="Não abre a janela interativa (útil em lote)")
    parser.add_argument("--saida", default=".",
                        help="Pasta de saída para as imagens salvas")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolução (DPI) da imagem salva")
    return parser.parse_args(argv)


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
        f'  python VC.py --coluna-x "{colunas[0]}" --coluna-y "{colunas[1]}" '
        f'--coluna-ciclo "Scan" --ciclo-inicial 3\n')


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = analisar_argumentos(argv)

    carregador = CarregadorDados(
        args.pasta, extensao=args.extensao, delimitador=args.delimitador)

    # Se --listar-colunas foi especificado, exibe as colunas e sai
    if args.listar_colunas:
        exibir_colunas_disponiveis(carregador)
        return

    if args.ciclo_exato and args.ciclo_inicial is None:
        raise SystemExit("Use --ciclo-inicial junto com --ciclo-exato.")

    arquivos = args.arquivo if args.arquivo else carregador.listar_arquivos()

    if not arquivos:
        print(
            f"Nenhum arquivo '{args.extensao}' encontrado em '{args.pasta}'.")
        return

    if args.comparativo and len(arquivos) < 2:
        print("Aviso: --comparativo com apenas um arquivo selecionado; "
              "gerando o gráfico normalmente.\n")

    painel = PainelVoltametria(
        coluna_x=args.coluna_x,
        coluna_y=args.coluna_y,
        coluna_ciclo=args.coluna_ciclo,
        rotulo_x=args.rotulo_x,
        rotulo_y=args.rotulo_y,
        unidade_y="A",
        colormap=args.colormap,
    )

    pasta_saida = Path(args.saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    if args.comparativo:
        try:
            dados = {nome: carregador.carregar(nome) for nome in arquivos}
            titulo = args.titulo or "Comparativo de Voltametria Cíclica"
            fig, _ = painel.gerar_comparativo(
                dados,
                titulo=titulo,
                ciclo_inicial=args.ciclo_inicial,
                ciclo_exato=args.ciclo_exato,
            )

            if not args.sem_salvar:
                caminho = pasta_saida / "VC_comparativo.png"
                PainelVoltametria.salvar(fig, caminho, dpi=args.dpi)
                print(f"Gráfico comparativo salvo como '{caminho}'")

            if not args.sem_exibir:
                plt.show()

            plt.close(fig)
        except ValueError as e:
            print("\nErro ao gerar o gráfico comparativo:")
            print(e)
            print(
                "\nDica: Use --listar-colunas para descobrir os nomes exatos das colunas.\n")
        return

    for nome_arquivo in arquivos:
        try:
            df = carregador.carregar(nome_arquivo)
            titulo = args.titulo or f"Voltametria Cíclica — {nome_arquivo}"
            fig, _ = painel.gerar(
                df,
                titulo=titulo,
                ciclo_inicial=args.ciclo_inicial,
                ciclo_exato=args.ciclo_exato,
                cor_por_ciclo=not args.sem_cor_por_ciclo,
            )

            if not args.sem_salvar:
                caminho = pasta_saida / f"VC_{Path(nome_arquivo).stem}.png"
                PainelVoltametria.salvar(fig, caminho, dpi=args.dpi)
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