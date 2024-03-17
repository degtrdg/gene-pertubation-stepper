"use client";

import { Combobox } from "@/components/ui/autocomplete";
import { submitForm } from "@/lib/action";
import { useEffect, useState } from "react";
import { useFormState, useFormStatus } from "react-dom";

interface FormComponentProps {
  proteinNames: string[];
  setInitialParams: (params: {
    geneName: string;
    perturbation: string;
    targetCondition: string;
  }) => void;
  setFormSuccess: (success: boolean) => void;
}

export default function FormComponent({
  proteinNames,
  setInitialParams,
  setFormSuccess,
}: FormComponentProps) {
  const initialState = { message: "" };
  const [state, formAction] = useFormState(submitForm, initialState);
  const [isDisabled, setIsDisabled] = useState(false);
  const { pending } = useFormStatus();

  useEffect(() => {
    if (state.message.startsWith("success")) {
      const successData = JSON.parse(state.message.substring(8)); // Extract JSON data from the message
      setInitialParams(successData);
      setFormSuccess(true);
      setIsDisabled(true);
    }
  }, [state.message]);

  return (
    <div className="flex flex-col p-4 bg-gray-50 rounded dark:bg-gray-800">
      <form
        action={formAction}
        className={`flex flex-col justify-between h-full ${
          isDisabled ? "pointer-events-none opacity-50" : ""
        }`}
      >
        <div className="space-y-4 flex-grow">
          <div className="flex flex-col">
            <label htmlFor="geneName" className="mb-2 font-semibold">
              Target Gene:
            </label>
            <Combobox proteinNames={proteinNames} disabled={isDisabled} />
          </div>
          <div className="flex flex-col">
            <label htmlFor="perturbation" className="mb-2 font-semibold">
              Perturbation:
            </label>
            <input
              type="text"
              name="perturbation"
              defaultValue={"knockout"}
              className="px-4 py-2 border rounded w-full"
              disabled={isDisabled}
            />
          </div>
          <div className="flex flex-col">
            <label htmlFor="targetCondition" className="mb-2 font-semibold">
              Target Condition:
            </label>
            <input
              type="text"
              name="targetCondition"
              defaultValue={"apoptosis"}
              className="px-4 py-2 border rounded w-full"
              disabled={isDisabled}
            />
          </div>
        </div>
        {!isDisabled && (
          <button
            type="submit"
            aria-disabled={pending}
            className="px-4 py-2 bg-blue-500 text-white font-bold rounded hover:bg-blue-700 disabled:opacity-50 w-full mt-4"
          >
            Submit
          </button>
        )}
      </form>
      {state.message && !state.message.includes("success") && (
        <p className="mt-4 text-red-500">{state.message}</p>
      )}
    </div>
  );
}
