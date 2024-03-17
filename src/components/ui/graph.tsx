"use client";
import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

interface Node {
  id: string;
  group?: number;
  fx?: number | null;
  fy?: number | null;
  x?: number;
  y?: number;
  data: { label: string };
}

interface Link {
  id: string;
  source: string | Node;
  target: string | Node;
  weight: string;
}

const width = 500; // SVG width
const height = 800; // SVG height

const GraphComponent = ({ nodes, edges }: { nodes: Node[]; edges: Link[] }) => {
  const d3Container = useRef(null);
  const [currentNodes, setCurrentNodes] = useState<Node[]>(nodes);
  const [currentEdges, setCurrentEdges] = useState<Link[]>(edges);

  // Effect to update state when props change
  useEffect(() => {
    setCurrentNodes(nodes);
    setCurrentEdges(edges);
  }, [nodes, edges]); // Re-run when nodes or edges props change

  useEffect(() => {
    if (d3Container.current) {
      const svg = d3.select(d3Container.current);
      svg.selectAll("*").remove();

      const simulation = d3
        .forceSimulation(currentNodes)
        .force(
          "link",
          d3
            .forceLink(currentEdges)
            .id((d: Node) => d.id)
            .distance(70)
        )
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2));

      const link = svg
        .append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(currentEdges)
        .enter()
        .append("line")
        .attr("stroke-width", 1)
        .attr("stroke", "black") // Set a stroke color to ensure visibility

        .on("mouseover", function (event, d) {
          d3.select(this).attr("stroke-width", 3);
          const weightText = `Weight: ${d.weight}`;
          svg
            .append("text")
            .attr("class", "weight-text")
            .attr("x", event.offsetX)
            .attr("y", event.offsetY - 10)
            .text(weightText);
        })
        .on("mouseout", function () {
          d3.select(this).attr("stroke-width", 1);
          svg.selectAll(".weight-text").remove();
        });

      const node = svg
        .append("g")
        .attr("class", "nodes")
        .selectAll("node")
        .data(currentNodes)
        .enter()
        .append("circle")
        .attr("r", 5)
        .attr("fill", "#1f77b4");

      // Define drag behaviors
      const drag = d3
        .drag<SVGCircleElement, Node>()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
          // Update the positions of the nodes and links immediately
          node.attr("cx", (n) => n.x).attr("cy", (n) => n.y);
          link
            .attr("x1", (l) => l.source.x)
            .attr("y1", (l) => l.source.y)
            .attr("x2", (l) => l.target.x)
            .attr("y2", (l) => l.target.y);
        })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        });

      // Apply the drag behavior to the nodes
      //   node.call(drag);

      const labels = svg
        .append("g")
        .attr("class", "labels")
        .selectAll("text")
        .data(currentNodes)
        .enter()
        .append("text")
        .text((d) => d.id)
        .attr("x", 8)
        .attr("y", ".31em");

      simulation.on("tick", () => {
        // Define padding
        const nodeRadius = 20; // Adjust if nodes have different sizes
        const labelPadding = 50; // Adjust based on average label size
        const padding = nodeRadius + labelPadding;

        // Constrain nodes to be within the bounds of the SVG
        node
          .attr("cx", (d) => Math.max(padding, Math.min(width - padding, d.x!)))
          .attr("cy", (d) =>
            Math.max(nodeRadius, Math.min(height - nodeRadius, d.y!))
          );

        link
          .attr("x1", (d) =>
            Math.max(padding, Math.min(width - padding, d.source.x!))
          )
          .attr("y1", (d) =>
            Math.max(nodeRadius, Math.min(height - nodeRadius, d.source.y!))
          )
          .attr("x2", (d) =>
            Math.max(padding, Math.min(width - padding, d.target.x!))
          )
          .attr("y2", (d) =>
            Math.max(nodeRadius, Math.min(height - nodeRadius, d.target.y!))
          );

        labels
          .attr(
            "x",
            (d) => Math.max(padding, Math.min(width - padding, d.x!)) + 10
          )
          .attr("y", (d) =>
            Math.max(nodeRadius, Math.min(height - nodeRadius, d.y!))
          );

        simulation.nodes(currentNodes);
        simulation.force<d3.ForceLink<Node, Link>>("link")?.links(currentEdges);
        simulation.alpha(1).restart();
      });
    }
  }, [currentNodes, currentEdges]); // Only re-run the effect if currentNodes or currentEdges change

  return (
    <svg
      className="d3-component"
      width={width}
      height={height}
      ref={d3Container}
    />
  );
};

export default GraphComponent;
