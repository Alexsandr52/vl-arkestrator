// Copies microphone samples (mono, context sample rate) to the page for WAV encoding.
class Recorder extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("recorder", Recorder);
