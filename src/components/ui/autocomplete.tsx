"use client";
import React, { useState, useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";

interface ComboboxProps {
  proteinNames: string[];
  disabled?: boolean;
}
export function Combobox({ proteinNames, disabled }: ComboboxProps) {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState("");
  const searchParams = useSearchParams();
  const { replace } = useRouter();

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };

    // Attach the event listener
    document.addEventListener("keydown", handleKeyDown);

    // Remove event listener on cleanup
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []); // Empty dependency array ensures this effect runs only once

  const handleSearch = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!disabled) {
      const term = event.target.value;
      setValue(term);
      const params = new URLSearchParams(searchParams.toString());
      if (term) {
        params.set("query", term);
        setOpen(true);
      } else {
        params.delete("query");
        setOpen(false);
      }
      replace(`?${params.toString()}`);
    }
  };

  const handleSelect = (protein: string) => {
    if (!disabled) {
      setValue(protein);
      setOpen(false);
      const params = new URLSearchParams(searchParams.toString());
      params.set("query", protein);
      replace(`?${params.toString()}`);
    }
  };

  // Close the dropdown if clicked outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        !disabled &&
        (event.target as Element).closest(".combobox-container") === null
      ) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [disabled]);

  // On start if there are query params, set the value
  useEffect(() => {
    const query = searchParams.get("query");
    if (query) {
      setValue(query);
    }
  }, []);

  return (
    <div className="combobox-container flex flex-col items-start relative">
      <input
        type="text"
        name="geneName"
        placeholder="Search protein..."
        className={`px-4 py-2 w-full border ${
          disabled ? "bg-gray-200 cursor-not-allowed" : ""
        }`}
        onChange={handleSearch}
        value={value}
        onFocus={() => !disabled && setOpen(true)}
        disabled={disabled}
      />
      {open && (
        <ul
          className="absolute z-10 mt-1 w-full max-h-60 overflow-auto border bg-white"
          style={{ top: "100%" }}
        >
          {proteinNames.map((protein) => (
            <li
              key={protein}
              className={`px-4 py-2 hover:bg-gray-100 cursor-pointer ${
                disabled ? "cursor-not-allowed" : ""
              }`}
              onClick={() => handleSelect(protein)}
            >
              {protein}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
