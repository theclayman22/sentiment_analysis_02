

### Benchmarks (ungefähr)
- **VADER**: ~1000 Texte/Minute (lokal)
- **HuggingFace**: ~100 Texte/Minute  
- **OpenAI/DeepSeek**: ~60 Texte/Minute
- **Emotion Arc**: ~1 Text/Minute (20 Segmente)

### Optimierungen
- Parallele Verarbeitung mit ThreadPoolExecutor
- Batch-Processing für große Datenmengen
- Caching für wiederholte Anfragen
- Intelligente API-Rotation

## 🤝 Contributing

1. Fork das Repository
2. Erstelle einen Feature-Branch (`git checkout -b feature/amazing-feature`)
3. Committe deine Änderungen (`git commit -m 'Add amazing feature'`)
4. Push zum Branch (`git push origin feature/amazing-feature`)
5. Öffne einen Pull Request

## 📝 Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe `LICENSE` für Details.

## ⚠️ Hinweise

- **API-Kosten**: OpenAI und DeepSeek APIs sind kostenpflichtig
- **Rate Limits**: Beachte die API-Limits der verschiedenen Anbieter
- **Datenverarbeitung**: Texte werden zur Analyse an externe APIs gesendet
- **Privacy**: Verwende keine sensiblen Daten ohne entsprechende Vorsichtsmaßnahmen

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/sentiment-analysis-toolkit/issues)
- **Dokumentation**: Siehe `/docs` Ordner
- **Wiki**: [GitHub Wiki](https://github.com/your-username/sentiment-analysis-toolkit/wiki)

---

**Entwickelt mit ❤️ für professionelle Sentiment-Analyse**
