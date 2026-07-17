# FAQ PDFs

Coloque aqui os arquivos `.pdf` de FAQ de renegociação de dívida. Cada PDF é lido como texto (sem OCR — arquivos precisam ter texto real, não apenas imagem escaneada), dividido em chunks e indexado no OpenSearch (`faq_chunks`) para busca vetorial via `GET /search`.

A ingestão roda automaticamente na subida do serviço (se `OPENAI_API_KEY` estiver configurada) e pode ser refeita a qualquer momento, sem reiniciar o container:

```bash
curl -X POST http://localhost:8500/admin/reindex
```

Arquivos já indexados (mesmo hash de conteúdo) são pulados automaticamente — só reprocessa o que for novo ou tiver mudado.
