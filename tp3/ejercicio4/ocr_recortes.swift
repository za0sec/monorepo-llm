import Foundation
import Vision

for ruta in CommandLine.arguments.dropFirst() {
    let solicitud = VNRecognizeTextRequest()
    solicitud.recognitionLevel = .accurate
    solicitud.recognitionLanguages = ["es-ES", "en-US"]
    solicitud.usesLanguageCorrection = false

    do {
        let imagen = VNImageRequestHandler(url: URL(fileURLWithPath: ruta))
        try imagen.perform([solicitud])
        let lineas = (solicitud.results ?? []).compactMap { $0.topCandidates(1).first?.string }
        let resultado = try JSONEncoder().encode(lineas)
        print(String(decoding: resultado, as: UTF8.self))
    } catch {
        let mensaje = "OCR falló en \(URL(fileURLWithPath: ruta).lastPathComponent): \(error)\n"
        FileHandle.standardError.write(Data(mensaje.utf8))
        exit(1)
    }
}
