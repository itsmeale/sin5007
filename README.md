# SIN5007 Reconhecimento de padrões

[Link para apresentação](https://docs.google.com/presentation/d/1y-lzXKcS2_UlgfjSSH7n2GR3K5IyFCK4GlZ10-nf4y4/edit#slide=id.p)
![Link para apresentação](assets/presentation/presentation.png)


## Estrutura do repositório
```
├── data
│   ├── preprocessed            -- Dados intermediários (pre-processados)
│   ├── results                 -- Resultados
│   └── raw                     -- Dados brutos
├── docs                        -- Documentações do projeto e datasets
├── notebooks                   -- Experimentos em jupyter notebooks
├── outputs                     -- Visualizações de dados
├── src                         -- Scripts utilizados
│   ├── dataviz                 -- Scripts de visualização de dados
│   ├── evaluation              -- Validação cruzada e métricas
│   ├── features                -- Remoção de outliers e train test split
│   ├── train                   -- Treinamento de modelos
│   └── toolbox                 -- Funções auxiliares
├── README.md
├── requirements.txt            -- Bibliotecas necessárias
└── setup.py                    -- Arquivo de configuração do python
```

## Como contribuir

### Linux
```
$ python3 -m venv venv
$ . venv/bin/activate
$ pip install -r requirements
```

## Contributors

@cabertoldi @igabid @ngalindojr @itsmeale