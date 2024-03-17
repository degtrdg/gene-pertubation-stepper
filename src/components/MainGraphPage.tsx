"use client";
import React, { useState, useEffect, use } from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import FormComponent from "./ui/custom-form";
import GraphComponent from "./ui/graph";
import clsx from "clsx";

interface MainGraphPageProps {
  proteinNames: string[];
  initialParams?: {
    geneName: string;
    perturbation: string;
    targetCondition: string;
  };
}

const MainGraphPage = ({ proteinNames }: MainGraphPageProps) => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [graphData, setGraphData] = useState<any | null>(null);
  const [streamingText, setStreamingText] = useState<string[]>([]);
  const [answer, setAnswer] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [formSuccess, setFormSuccess] = useState(false);
  const [formValues, setFormValues] = useState({
    geneName: "",
    perturbation: "",
    targetCondition: "",
  });
  const [loadingDots, setLoadingDots] = useState(".");

  useEffect(() => {
    const intervalId = setInterval(() => {
      setLoadingDots((prevDots) =>
        prevDots.length < 3 ? prevDots + "." : "."
      );
    }, 500); // Adjust the interval time as needed

    return () => clearInterval(intervalId); // Clear the interval on component unmount
  }, []);

  // Determine the base URL based on the environment
  const baseUrl =
    process.env.NODE_ENV === "development"
      ? "http://localhost:8000"
      : "https://your-production-url.com";

  const startStreamingText = (sessionId: string) => {
    setIsLoading(true); // Start loading when streaming begins

    // Construct the URL with the session ID
    const url = new URL(`/api/start/${encodeURIComponent(sessionId)}`, baseUrl);

    // Initialize a new EventSource
    const eventSource = new EventSource(url.toString());

    eventSource.onmessage = (event) => {
      if (event.data === "[DONE]") {
        setIsLoading(false);
        setStreamingText((prevText) => [
          ...prevText,
          "<br /> <br /> <br /> ---  <br /> <br /> <br />",
        ]);
        eventSource.close();
      } else {
        try {
          const parsedData = event.data;
          // Decode the newline characters
          const decodedData = parsedData.replace(/\\m/g, "\n");
          setStreamingText((prevText) => [...prevText, decodedData]);
        } catch (error) {
          console.error("Failed to parse event data:", event.data, error);
        }
      }
    };

    eventSource.onerror = (error) => {
      console.error("EventSource failed:", error);
      setIsLoading(false); // Stop loading on error
      eventSource.close(); // Close the connection on error
    };
  };

  const streamSimulationChange = (sessionId: string, next_protein: string) => {
    setIsLoading(true); // Start loading when streaming begins
    const url = new URL(
      `/api/simulate_change/${encodeURIComponent(
        sessionId
      )}/${encodeURIComponent(next_protein)}`,
      baseUrl
    );
    const eventSource = new EventSource(url.toString());
    // Add a newline to the streaming text
    setStreamingText((prevText) => [...prevText, ""]);

    eventSource.onmessage = (event) => {
      if (event.data === "[DONE]") {
        setIsLoading(false);
        setStreamingText((prevText) => [
          ...prevText,
          "<br /> <br /> <br /> ---  <br /> <br /> <br />",
        ]);
        eventSource.close();
      } else {
        try {
          const parsedData = event.data;
          // Decode the newline characters
          const decodedData = parsedData.replace(/\\m/g, "\n");
          setStreamingText((prevText) => [...prevText, decodedData]);
        } catch (error) {
          console.error("Failed to parse event data:", event.data, error);
        }
      }
    };

    eventSource.onerror = (error) => {
      console.error("EventSource failed:", error);
      setIsLoading(false); // Stop loading on error
      eventSource.close(); // Close the connection on error
    };
  };

  // Function to handle the "Next" button click
  const handleNext = async () => {
    console.log("Next button clicked");
    if (sessionId && !isLoading) {
      setIsLoading(true); // Start loading
      try {
        // Fetch the graph data
        const graphResponse = await fetch(
          `${baseUrl}/api/visualize`, // Corrected endpoint
          {
            method: "POST", // Changed to POST
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ session_id: sessionId }), // Send session_id in the body
          }
        );
        if (!graphResponse.ok) {
          throw new Error(`HTTP error! status: ${graphResponse.status}`);
        }
        const graphData = await graphResponse.json();

        setGraphData({
          nodes: graphData.nodes,
          edges: graphData.edges,
        });
        // Start streaming the simulation changes
        streamSimulationChange(sessionId, graphData.next_protein);
        // Fetch the answer for the condition check
        const answerResponse = await fetch(
          `${baseUrl}/api/check_condition`, // Corrected endpoint
          {
            method: "POST", // Changed to POST
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ session_id: sessionId }), // Send session_id in the body
          }
        );
        if (!answerResponse.ok) {
          throw new Error(`HTTP error! status: ${answerResponse.status}`);
        }
        const answerData = await answerResponse.json();
        console.log("Answer data:", answerData);
        setAnswer(answerData.condition_met);
      } catch (error) {
        console.error("There was a problem with the fetch operation:", error);
      } finally {
        setIsLoading(false); // Stop loading regardless of the outcome
      }
    }
  };

  useEffect(() => {
    // Function to initialize the protein explorer
    const initializeProteinExplorer = async (
      geneName: string,
      perturbation: string,
      targetCondition: string
    ) => {
      if (!geneName || !perturbation || !targetCondition) {
        console.error("Missing required fields to initialize protein explorer");
        return;
      }
      if (isLoading) return;
      setIsLoading(true);
      try {
        const response = await fetch(`${baseUrl}/api/initialize`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            start_protein: geneName,
            perturbation: perturbation,
            target_condition: targetCondition,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data && data.nodes && data.edges && data.session_id) {
          setGraphData({
            nodes: data.nodes,
            edges: data.edges,
          });
          setSessionId(data.session_id);
          startStreamingText(data.session_id);
        } else {
          console.error(
            "Graph data from API is not in the expected format:",
            data
          );
        }
      } catch (error) {
        console.error("There was a problem with the fetch operation:", error);
      } finally {
        setIsLoading(false);
      }
    };
    if (
      formValues.geneName &&
      formValues.perturbation &&
      formValues.targetCondition
    ) {
      initializeProteinExplorer(
        formValues.geneName,
        formValues.perturbation,
        formValues.targetCondition
      );
    }
  }, [formValues]);

  const preprocessForMarkdown = (textArray: string[]) => {
    return textArray
      .map((text) => {
        // Replace single newlines with <br> tags to maintain line breaks
        return text.replace(/\n/g, "<br />\n");
      })
      .join("");
  };

  // Add a custom CSS class for mobile layout
  const mobileLayoutClass = "flex flex-col w-full";
  // State to track if the layout is for mobile
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    function handleResize() {
      setIsMobile(window.innerWidth < 768);
    }

    window.addEventListener("resize", handleResize);
    handleResize(); // Set the initial value

    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div
      className={clsx("flex flex-row h-screen", {
        [mobileLayoutClass]: isMobile,
      })}
    >
      <aside
        className={clsx("w-1/6 min-w-", { "w-full": isMobile })}
        aria-label="Sidebar"
      >
        <div className="overflow-y-auto py-4 px-3 bg-gray-50 rounded dark:bg-gray-800 h-full my-4">
          <FormComponent
            proteinNames={proteinNames}
            setInitialParams={setFormValues}
            setFormSuccess={setFormSuccess}
          />
          {formSuccess && (
            <button
              onClick={handleNext}
              className={`mt-4 w-full font-bold py-2 px-4 rounded ${
                isLoading
                  ? "bg-gray-500 text-gray-300 cursor-not-allowed"
                  : "bg-blue-500 text-white hover:bg-blue-700"
              }`}
              disabled={isLoading}
            >
              Next
            </button>
          )}

          {streamingText && (
            <p
              style={{
                color: answer ? "green" : "red",
              }}
              className="mt-4"
            >
              {`Condition ${formValues?.targetCondition} has been reached?: ${answer}`}
            </p>
          )}
        </div>
      </aside>
      <div
        className={clsx("w-1/2", { "w-full": isMobile })}
        style={{ minWidth: 0 }}
      >
        <div className="h-full bg-gray-200 overflow-auto">
          <div
            className={clsx(
              "streaming-text-area p-5 h-full bg-gray-200 overflow-auto",
              {
                "flex justify-center items-center":
                  formSuccess && !streamingText.length,
              }
            )}
          >
            {formSuccess && !streamingText.length ? (
              <div className="text-lg font-semibold">
                <div>{`Takes a while, sorry :(`}</div>
                <div>Loading{loadingDots}</div>
              </div>
            ) : (
              <ReactMarkdown rehypePlugins={[rehypeRaw]}>
                {preprocessForMarkdown(streamingText)}
              </ReactMarkdown>
            )}
          </div>
        </div>
      </div>
      <div
        id="graph-area"
        className={clsx("w-2/5 p-4 flex justify-center items-center", {
          "w-full": isMobile,
        })}
      >
        {formSuccess && !graphData ? (
          <div className="text-lg font-semibold">
            <div>The server takes a while to load the data</div>
            <div>Loading{loadingDots}</div>
          </div>
        ) : (
          graphData && (
            <GraphComponent nodes={graphData.nodes} edges={graphData.edges} />
          )
        )}
      </div>
    </div>
  );
};

export default MainGraphPage;
