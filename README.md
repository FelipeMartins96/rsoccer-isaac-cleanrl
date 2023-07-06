# rsoccer-isaac-cleanrl

TODO:

    Notas:
    - Testar apenas com falta de ataque inicialmente

- [ ] Identificar como estado terminal
    - [ ] Falta de ataque
    - [ ] Falta de defesa
- [ ] Recompensa
    - [ ] Penalização em caso de falta
    - [ ] Separar os dois casos para poder acompanhar o desempenho de cada um
    - [ ] Ajustar wrapper de recompensas para as novas recompensas
OPT:
- [ ] Observações
    - [ ] Flags de dentro da area de ataque bola e robos
    - [ ] Flags de dentro da area de defesa bola e robos
    - [ ] Ajustar tamanho das observações
    - [ ] Como deixar compativel com as redes anteriores

Refatorar:
 - Unir pesos e done para faltas
 - Retornar flag de motivo de done
 - Logar motivo de done na validação

Avaliar:
 - Validar com faltas do amarelo