import Benchmark from "@/components/Benchmark";
import Citation from "@/components/Citation";
import Collection from "@/components/Collection";
import DataAccess from "@/components/DataAccess";
import Findings from "@/components/Findings";
import Footer from "@/components/Footer";
import Hero from "@/components/Hero";
import Method from "@/components/Method";
import Results from "@/components/Results";
import Reveal from "@/components/Reveal";
import Species from "@/components/Species";
import TryTask from "@/components/TryTask";

export default function Page() {
  return (
    <>
      <Hero />
      <main>
        <Reveal>
          <TryTask />
        </Reveal>
        <Reveal>
          <Species />
        </Reveal>
        <Reveal>
          <Collection />
        </Reveal>
        <Reveal>
          <Method />
        </Reveal>
        <Reveal>
          <Benchmark />
        </Reveal>
        <Reveal>
          <Results />
        </Reveal>
        <Reveal>
          <Findings />
        </Reveal>
        <Reveal>
          <DataAccess />
        </Reveal>
        <Reveal>
          <Citation />
        </Reveal>
      </main>
      <Footer />
    </>
  );
}
