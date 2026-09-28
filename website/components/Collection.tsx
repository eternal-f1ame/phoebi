import { leads, sections } from "@/content";
import CultureFlow from "./CultureFlow";
import SectionHeading from "./SectionHeading";

export default function Collection() {
  return (
    <section id="collection" className="section">
      <div className="container">
        <SectionHeading {...sections.collection} />
        <p className="lead">{leads.collection}</p>
        <CultureFlow />
      </div>
    </section>
  );
}
