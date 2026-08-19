#!/usr/bin/env python3
"""
AGRUPAR_CA.py (versão OOP)

Agrupa arquivos de cronoamperometria (CA) exportados em pares (pulso base
= potencial 0 e pulso aplicado = potencial != 0), unindo cada par num
único arquivo de saída.

COMO O SCRIPT IDENTIFICA OS PARES
----------------------------------
Os arquivos precisam terminar com um sufixo numérico "_1" ou "_2" antes da
extensão, por exemplo:

    CA_012V_grupo1_1.txt   -> chave de agrupamento: "CA_012V_grupo1"
    CA_012V_grupo1_2.txt   -> chave de agrupamento: "CA_012V_grupo1"

Todo o texto antes de "_<numero>" é usado como chave. Arquivos com a mesma
chave são agrupados. Dentro de cada grupo, os arquivos são ordenados pelo
número do sufixo (_1 antes de _2) e concatenados em sequência. O
cabeçalho (primeira linha) é escrito uma única vez.

USO
---
    python AGRUPAR_CA.py [pasta] [opções]

Exemplos:
    python AGRUPAR_CA.py
    python AGRUPAR_CA.py ./dados
    python AGRUPAR_CA.py ./dados -e .csv -o resultado
    python AGRUPAR_CA.py ./dados --padrao "^(.*)_pulso(\\d+)$"

Rode "python AGRUPAR_CA.py -h" para ver todas as opções.

Os arquivos de saída são salvos, por padrão, numa subpasta "agrupados/"
dentro da pasta de entrada, com o nome "<chave>.txt".
"""

import argparse
import re
from pathlib import Path
from dataclasses import dataclass, field


# Regex padrão que captura "<chave>_<numero>" a partir do nome sem extensão.
# Pode ser sobrescrito pelo argumento --padrao no terminal.
PADRAO_SUFIXO_DEFAULT = r"^(.*)_0*(\d+)$"


@dataclass
class ArquivoCA:
    """Representa um único arquivo de dados (um pulso) já lido do disco."""

    caminho: Path
    indice: int
    cabecalho: str = ""
    linhas_dados: list[str] = field(default_factory=list)

    @classmethod
    def carregar(cls, caminho: Path, indice: int) -> "ArquivoCA":
        with open(caminho, "r", encoding="utf-8-sig") as f:
            linhas = [l.rstrip("\n\r") for l in f if l.strip() != ""]
        cabecalho = linhas[0] if linhas else ""
        dados = linhas[1:] if linhas else []
        return cls(caminho=caminho, indice=indice, cabecalho=cabecalho, linhas_dados=dados)

    @property
    def n_linhas(self) -> int:
        return len(self.linhas_dados)


class GrupoCA:
    """Agrupa vários ArquivoCA que pertencem à mesma medida (mesma chave)."""

    def __init__(self, chave: str):
        self.chave = chave
        self._arquivos: dict[int, ArquivoCA] = {}

    def adicionar(self, arquivo: ArquivoCA) -> None:
        self._arquivos[arquivo.indice] = arquivo

    @property
    def indices_ordenados(self) -> list[int]:
        return sorted(self._arquivos)

    @property
    def nomes_origem(self) -> str:
        return ", ".join(self._arquivos[i].caminho.name for i in self.indices_ordenados)

    def cabecalho(self) -> str:
        primeiro = self._arquivos[self.indices_ordenados[0]]
        return primeiro.cabecalho

    def linhas_concatenadas(self) -> list[str]:
        linhas: list[str] = []
        for indice in self.indices_ordenados:
            linhas.extend(self._arquivos[indice].linhas_dados)
        return linhas

    def escrever(self, pasta_saida: Path, extensao: str) -> Path:
        caminho_saida = pasta_saida / f"{self.chave}{extensao}"
        linhas = self.linhas_concatenadas()
        with open(caminho_saida, "w", encoding="utf-8", newline="\n") as f:
            f.write(self.cabecalho() + "\n")
            f.write("\n".join(linhas) + "\n")
        return caminho_saida


class AgrupadorCronoamperometria:
    """
    Varre uma pasta, identifica pares/grupos de arquivos de cronoamperometria
    pelo prefixo comum do nome e escreve um arquivo unido por grupo.
    """

    def __init__(
        self,
        pasta_entrada: Path,
        extensao: str = ".txt",
        subpasta_saida: str = "agrupados",
        padrao_sufixo: str = PADRAO_SUFIXO_DEFAULT,
    ):
        self.pasta_entrada = pasta_entrada.resolve()
        self.extensao = extensao
        self.pasta_saida = self.pasta_entrada / subpasta_saida
        self.padrao_sufixo = re.compile(padrao_sufixo)
        self.grupos: dict[str, GrupoCA] = {}
        self.ignorados: list[str] = []

    def identificar_grupo(self, caminho: Path) -> tuple[str | None, int | None]:
        """Extrai (chave, indice) do nome do arquivo, ou (None, None) se não bater com o padrão."""
        m = self.padrao_sufixo.match(caminho.stem)
        if not m:
            return None, None
        chave, indice_str = m.groups()
        return chave, int(indice_str)

    def descobrir_arquivos(self) -> list[Path]:
        return sorted(self.pasta_entrada.glob(f"*{self.extensao}"))

    def classificar(self) -> None:
        for caminho in self.descobrir_arquivos():
            chave, indice = self.identificar_grupo(caminho)
            if chave is None:
                self.ignorados.append(caminho.name)
                continue
            arquivo = ArquivoCA.carregar(caminho, indice)
            self.grupos.setdefault(chave, GrupoCA(chave)).adicionar(arquivo)

    def executar(self) -> None:
        self.classificar()

        if not self.grupos and not self.ignorados:
            print(f"Nenhum arquivo {self.extensao} encontrado em: {self.pasta_entrada}")
            return

        if self.ignorados:
            print("Arquivos ignorados (nome não segue o padrão '<chave>_<numero>.txt'):")
            for nome in self.ignorados:
                print(f"  - {nome}")
            print()

        self.pasta_saida.mkdir(exist_ok=True)

        for chave in sorted(self.grupos):
            grupo = self.grupos[chave]
            caminho_saida = grupo.escrever(self.pasta_saida, self.extensao)
            n_linhas = len(grupo.linhas_concatenadas())
            print(f"[OK] {caminho_saida.name}  <-  {grupo.nomes_origem}  ({n_linhas} linhas de dados)")

        print(f"\nArquivos agrupados salvos em: {self.pasta_saida}")


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="AGRUPAR_CA.py",
        description=(
            "Agrupa arquivos de cronoamperometria exportados em pares "
            "(ex.: CA_004_1.txt / CA_004_2.txt) em um único arquivo por medida."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pasta",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Pasta onde estão os arquivos de entrada.",
    )
    parser.add_argument(
        "-e", "--extensao",
        default=".txt",
        help="Extensão dos arquivos de dados a procurar (inclua o ponto).",
    )
    parser.add_argument(
        "-o", "--saida",
        default="agrupados",
        metavar="SUBPASTA",
        help="Nome da subpasta de saída, criada dentro da pasta de entrada.",
    )
    parser.add_argument(
        "--padrao",
        default=PADRAO_SUFIXO_DEFAULT,
        metavar="REGEX",
        help=(
            "Expressão regular usada para separar '<chave>' e '<numero>' no "
            "nome do arquivo (sem extensão). Deve conter exatamente dois "
            "grupos de captura: o primeiro é a chave de agrupamento e o "
            "segundo é o número usado para ordenar os arquivos dentro do "
            "grupo."
        ),
    )
    return parser


def main() -> None:
    parser = criar_parser()
    args = parser.parse_args()

    agrupador = AgrupadorCronoamperometria(
        pasta_entrada=args.pasta,
        extensao=args.extensao,
        subpasta_saida=args.saida,
        padrao_sufixo=args.padrao,
    )
    agrupador.executar()


if __name__ == "__main__":
    main()