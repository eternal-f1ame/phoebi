import { leads, sections } from "@/content";
import DecoderFlow from "./DecoderFlow";
import SectionHeading from "./SectionHeading";

export default function Method() {
  return (
    <section id="method" className="section">
      <div className="container">
        <SectionHeading {...sections.method} />
        <p className="lead">{leads.method}</p>
        <DecoderFlow />
      </div>
    </section>
  );
}
